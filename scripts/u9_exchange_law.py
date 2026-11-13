"""U9: an exchange-rate law and robustness of k* (needs u6 and u7).

(1) Fit k*(n) = K n / (n + n0) to the exchange rate of the confirmatory split against archive size n (least squares on
    log k*), on the point estimates and on each of the 1,000 bootstrap replicates of u7 (replicates with an infinite
    k* at any size are dropped and counted). Under a bias-variance reading, archive error ~ b^2 + s^2/n and
    experiment error ~ s_e^2/k, so k* saturates at K ~ s_e^2/b^2, set by the archive's bias.
    Out-of-sample check: predict k* for the archives of the two smaller splits (their training folds) and compare
    with the measured values and intervals.
(2) k* under three accuracy measures (significant pairs, all within-cell pairs, click-through lift), from fold
    means, and the slope of the experiments-only curve at k* (accuracy points per doubling of k).
Usage: python scripts/u9_exchange_law.py
Output: results/tables/u9_exchange_law.csv, results/tables/u9_kstar_metrics.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from newspop.data import ROOT

TAB = ROOT / "results" / "tables"
N_ALL = {"confirmatory": 18194}


def kstar(ks, e, target):
    lk = np.log(ks)
    for i in range(len(ks)):
        if e[i] >= target:
            if i == 0:
                return float(ks[0])
            f = (target - e[i - 1]) / (e[i] - e[i - 1])
            return float(np.exp(lk[i - 1] + f * (lk[i] - lk[i - 1])))
    return np.inf


def fit(n, k):
    r = least_squares(lambda p: np.log(np.exp(p[0]) * n / (n + np.exp(p[1]))) - np.log(k), x0=[np.log(1000), np.log(2000)])
    return np.exp(r.x)


def archive_size(split):
    """Mean number of training tests per fold = size of the full archive (one headline per test)."""
    runs = pd.read_csv(TAB / f"u6_runs_{split}.csv")
    from newspop import upw_rank as ur
    from sklearn.model_selection import GroupKFold
    S = ur.setup(split)
    return float(np.mean([len(tr) for tr, _ in GroupKFold(5).split(np.zeros(len(S.tests)), groups=S.article)])), runs


law = []
X = pd.read_csv(TAB / "u7_exchange_rate.csv")
fb = np.load(TAB / "u7_kstar_boot_confirmatory.npz")
keys = [tuple(k.split("|")) for k in fb["keys"]]
sizes_other = {sp: archive_size(sp)[0] for sp in ["exploratory", "holdout"]}
for arch in ["random", "published"]:
    sel = [(i, int(k[2])) for i, k in enumerate(keys) if k[0] == "O" and k[1] == arch]
    n = np.array([N_ALL["confirmatory"] if s == -1 else s for _, s in sel], float)
    Kb = fb["kstar"][[i for i, _ in sel]]                       # sizes x B
    x = X.query("split == 'confirmatory' and method == 'O' and archive == @arch")
    k0 = np.array([x.loc[x["size"] == s, "kstar"].iloc[0] for _, s in sel])
    K, n0 = fit(n, k0)
    ok = np.isfinite(Kb).all(0)
    P = np.array([fit(n, Kb[:, b]) for b in np.where(ok)[0]])
    row = dict(archive=arch, K=K, K_lo=np.quantile(P[:, 0], .025), K_hi=np.quantile(P[:, 0], .975), n0=n0,
               n0_lo=np.quantile(P[:, 1], .025), n0_hi=np.quantile(P[:, 1], .975), boot_dropped=int((~ok).sum()))
    for s_, kk in zip(n, k0):
        row[f"fit_{int(s_)}"] = K * s_ / (s_ + n0)
        row[f"obs_{int(s_)}"] = kk
    for sp, ns in sizes_other.items():
        pred = P[:, 0] * ns / (ns + P[:, 1])
        m = X.query("split == @sp and method == 'O' and archive == @arch and size == -1").iloc[0]
        row.update({f"{sp}_n": ns, f"{sp}_pred": K * ns / (ns + n0), f"{sp}_pred_lo": np.quantile(pred, .025),
                    f"{sp}_pred_hi": np.quantile(pred, .975), f"{sp}_obs": m.kstar, f"{sp}_obs_lo": m.kstar_lo,
                    f"{sp}_obs_hi": m.kstar_hi})
        for s in (1000, 2000):
            mm = X.query("split == @sp and method == 'O' and archive == @arch and size == @s").iloc[0]
            row[f"{sp}_{s}_pred"] = K * s / (s + n0)
            row[f"{sp}_{s}_obs"] = mm.kstar
    law.append(row)
L = pd.DataFrame(law).round(1)
L.to_csv(TAB / "u9_exchange_law.csv", index=False)
print(L.T.to_string())

out = []
for sp in ["confirmatory", "exploratory", "holdout"]:
    R = pd.read_csv(TAB / f"u6_runs_{sp}.csv")
    for met in ["within_acc_sig", "within_acc", "lift_pct"]:
        g = R.groupby(["method", "archive", "size", "k"])[met].mean()
        E = g.loc["E", "-", 0]
        ks = [k for k in sorted(E.index) if k > 0]
        e = np.array([E[k] for k in ks])
        for method in ["O", "O-raw"]:
            for arch in ["random", "published"]:
                for s in sorted(set(R.loc[(R.method == method) & (R.archive == arch), "size"])):
                    t = g.loc[method, arch, s, 0]
                    k_ = kstar(ks, e, t)
                    i = int(np.searchsorted(ks, k_)) if np.isfinite(k_) else len(ks) - 1
                    i = min(max(i, 1), len(ks) - 1)
                    slope = (e[i] - e[i - 1]) / np.log2(ks[i] / ks[i - 1])
                    out.append(dict(split=sp, metric=met, method=method, archive=arch, size=s, target=t, kstar=k_,
                                    slope_per_doubling=slope))
K_ = pd.DataFrame(out).round(4)
K_.to_csv(TAB / "u9_kstar_metrics.csv", index=False)
print(K_.query("size == -1").pivot_table(index=["split", "method", "archive"], columns="metric", values="kstar").round(0))
print(K_.query("size == -1 and metric == 'within_acc_sig'")[["split", "method", "archive", "slope_per_doubling"]])

# (3) archive-guided experiment selection (u8): each strategy minus random (mean of 3 draws), per learner and k,
# with article-cluster bootstrap intervals from the per-pair correctness.
act = []
for sp in ["confirmatory", "exploratory", "holdout"]:
    path = TAB / f"u8_pairs_{sp}.npz"
    if not path.exists():
        continue
    f = np.load(path)
    meta = pd.DataFrame([c.split("|") for c in f["configs"]], columns=["learner", "strategy", "draw", "k"])
    meta[["draw", "k"]] = meta[["draw", "k"]].astype(int)
    C, art = f["correct"], f["article"]
    labs, inv = np.unique(art, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=c, minlength=len(labs)) for c in C])
    cnt = np.bincount(inv, minlength=len(labs))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs)) for _ in range(1000)])
    acc0, accb = sums.sum(1) / cnt.sum(), (sums @ W.T) / (cnt @ W.T)
    for (learner, k), g in meta[meta.k > 0].groupby(["learner", "k"]):
        r = g.index[g.strategy == "random"]
        for st in ["margin-", "margin+", "committee", "design"]:
            i = g.index[g.strategy == st]
            d = accb[i].mean(0) - accb[r].mean(0)
            act.append(dict(split=sp, learner=learner, k=k, strategy=st, acc=acc0[i].mean(),
                            minus_random=acc0[i].mean() - acc0[r].mean(), lo=np.quantile(d, .025),
                            hi=np.quantile(d, .975)))
if act:
    A_ = pd.DataFrame(act).round(4)
    A_.to_csv(TAB / "u9_active.csv", index=False)
    print(A_.query("learner != 'PRIOR'").pivot_table(index=["split", "learner", "k"], columns="strategy",
                                                      values="minus_random").round(3).to_string())
