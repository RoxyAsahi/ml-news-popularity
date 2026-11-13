"""U7: the exchange-rate curve from u6, with intervals that include training variability.

k*(archive, size) = the number of randomized tests at which the experiments-only learner reaches the decision
accuracy of an archive model (log-linear interpolation between budgets). Each of B = 1,000 replicates resamples
held-out articles (cluster bootstrap) and picks, for every budget and every archive size, one of the three
training draws at random; the point estimate averages the draws.
Also: accuracy and click-through lift per configuration, and differences PRIOR - E and PRIOR-fixed - E.
Usage: python scripts/u7_exchange_curve.py [split ...]
Output: results/tables/u7_exchange_rate.csv, results/tables/u7_fusion.csv, u7_kstar_boot_<split>.npz (replicates)
"""
import sys

import numpy as np
import pandas as pd

from newspop.data import ROOT

TAB = ROOT / "results" / "tables"
splits = sys.argv[1:] or ["confirmatory", "exploratory", "holdout"]
B = 1000


def kstar(ks, e, target):
    lk = np.log(ks)
    for i in range(len(ks)):
        if e[i] >= target:
            if i == 0:
                return float(ks[0])
            f = (target - e[i - 1]) / (e[i] - e[i - 1])
            return float(np.exp(lk[i - 1] + f * (lk[i] - lk[i - 1])))
    return np.inf


ex_rows, fu_rows = [], []
for split in splits:
    try:
        f = np.load(TAB / f"u6_pairs_{split}.npz")
    except FileNotFoundError:
        continue
    meta = pd.DataFrame([c.split("|") for c in f["configs"]], columns=["method", "archive", "size", "draw", "k"])
    meta[["size", "draw", "k"]] = meta[["size", "draw", "k"]].astype(int)
    C, art = f["correct"], f["article"]
    labs, inv = np.unique(art, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=c, minlength=len(labs)) for c in C])
    cnt = np.bincount(inv, minlength=len(labs))
    runs = pd.read_csv(TAB / f"u6_runs_{split}.csv")
    lift = runs.groupby(["method", "archive", "size", "draw", "k"]).lift_pct.mean()
    ks = sorted(meta.k[(meta.method == "E") & (meta.k > 0)].unique())
    idx = {tuple(r): i for i, r in enumerate(meta[["method", "archive", "size", "draw", "k"]].itertuples(index=False))}
    archives = sorted({(a, s) for m, a, s in zip(meta.method, meta.archive, meta["size"]) if m in ("O", "O-raw")})
    rng = np.random.default_rng(0)

    def acc_of(w):
        return (sums @ w) / (cnt @ w)

    def curve(acc, pick):
        return np.array([acc[idx[("E", "-", 0, pick(k), k)]] for k in ks])

    def archive_acc(acc, method, a, s, pick):
        draws = sorted(meta.draw[(meta.method == method) & (meta.archive == a) & (meta["size"] == s)])
        return acc[idx[(method, a, s, pick(draws), 0)]]

    a0 = acc_of(np.ones(len(labs)))
    e0 = np.array([np.mean([a0[idx[("E", "-", 0, r, k)]] for r in range(3)]) for k in ks])
    boots = {}
    for b in range(B):
        w = np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs))
        acc = acc_of(w)
        e = curve(acc, lambda k: rng.integers(0, 3))
        for method in ["O", "O-raw"]:
            for a, s in archives:
                if (method, a, s, 0, 0) not in idx:
                    continue
                t = archive_acc(acc, method, a, s, lambda ds: ds[rng.integers(0, len(ds))])
                boots.setdefault((method, a, s), []).append((t, kstar(ks, e, t)))
    np.savez_compressed(TAB / f"u7_kstar_boot_{split}.npz", keys=np.array(["|".join(map(str, k)) for k in boots]),
                        kstar=np.array([[x[1] for x in v] for v in boots.values()]),
                        acc=np.array([[x[0] for x in v] for v in boots.values()]))
    for (method, a, s), v in boots.items():
        v = np.array(v)
        draws = sorted(meta.draw[(meta.method == method) & (meta.archive == a) & (meta["size"] == s)])
        t0 = np.mean([a0[idx[(method, a, s, d, 0)]] for d in draws])
        k0 = kstar(ks, e0, t0)
        q = np.quantile(v[:, 1], [.025, .975], method="inverted_cdf")
        ex_rows.append(dict(split=split, method=method, archive=a, size=s, acc=t0,
                            acc_lo=np.quantile(v[:, 0], .025), acc_hi=np.quantile(v[:, 0], .975),
                            kstar=k0, kstar_lo=q[0], kstar_hi=q[1],
                            lift_pct=np.mean([lift[(method, a, s, d, 0)] for d in draws])))
    # fusion: accuracy and lift by budget, differences to experiments only (draw-matched)
    for k in ks + [-1]:
        reps = sorted(meta.draw[(meta.method == "E") & (meta.k == k)])
        for method, a in [("E", "-"), ("PRIOR", "random"), ("PRIOR", "published"), ("PRIOR-fixed", "random"),
                          ("PRIOR-fixed", "published")]:
            if (method, a, -1 if method != "E" else 0, reps[0], k) not in idx:
                continue
            s_ = -1 if method != "E" else 0
            ii = [idx[(method, a, s_, r, k)] for r in reps]
            jj = [idx[("E", "-", 0, r, k)] for r in reps]
            W = np.stack([np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs)) for _ in range(300)])
            accb = (sums[ii] @ W.T) / (cnt @ W.T)
            eb = (sums[jj] @ W.T) / (cnt @ W.T)
            d = (accb - eb).mean(0)
            fu_rows.append(dict(split=split, method=method, archive=a, k=k, acc=np.mean(a0[ii]),
                                minus_E=np.mean(a0[ii]) - np.mean(a0[jj]), minus_E_lo=np.quantile(d, .025),
                                minus_E_hi=np.quantile(d, .975),
                                lift_pct=np.mean([lift[(method, a, s_, r, k)] for r in reps])))
    print(split, "done", flush=True)

X = pd.DataFrame(ex_rows).round(4)
X.to_csv(TAB / "u7_exchange_rate.csv", index=False)
F = pd.DataFrame(fu_rows).round(4)
F.to_csv(TAB / "u7_fusion.csv", index=False)
print(X.to_string(index=False))
print(F.to_string(index=False))
