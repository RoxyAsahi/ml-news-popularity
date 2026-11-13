"""U4: uncertainty for u3. How many randomized tests is the observational archive worth, with CIs?

From the per-pair correctness saved by u3 (significant within-test pairs of held-out articles), resample
articles (cluster bootstrap, B = 1000) and recompute, in each replicate:
  acc(config) for every method, budget k and repetition (repetitions averaged);
  k*(archive model) = the budget at which the experiments-only learner E(k) reaches the archive model's accuracy
    (linear interpolation in log k; capped at the largest budget);
  differences PRIOR(k) - E(k) and PRIOR(k) - O.
Usage: python scripts/u4_archive_worth.py [split ...]
Output: results/tables/u4_archive_worth.csv, results/tables/u4_curves.csv
"""
import sys

import numpy as np
import pandas as pd

from newspop.data import ROOT

TAB = ROOT / "results" / "tables"
splits = sys.argv[1:] or ["exploratory", "confirmatory", "holdout"]
B = 1000


def kstar(ks, e, target):
    """Smallest budget where the E curve reaches target (log-linear interpolation); inf if never."""
    lk = np.log(ks)
    for i in range(len(ks)):
        if e[i] >= target:
            if i == 0:
                return ks[0]
            f = (target - e[i - 1]) / (e[i] - e[i - 1])
            return float(np.exp(lk[i - 1] + f * (lk[i] - lk[i - 1])))
    return np.inf


worth, curves = [], []
for split in splits:
    f = np.load(TAB / f"u3_pairs_{split}.npz")
    cfg = [c.split("|") for c in f["configs"]]
    C, art = f["correct"], f["article"]
    meta = pd.DataFrame(cfg, columns=["method", "k", "rep"]).astype({"k": int, "rep": int})
    labs, inv = np.unique(art, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=C[i], minlength=len(labs)) for i in range(len(C))])
    cnt = np.bincount(inv, minlength=len(labs))
    ks = sorted(meta.k[(meta.method == "E") & (meta.k > 0)].unique())
    rng = np.random.default_rng(0)

    def stats(w):
        acc = (sums @ w) / (cnt @ w)
        a = meta.assign(acc=acc).groupby(["method", "k"]).acc.mean()
        e = np.array([a[("E", k)] for k in ks])
        out = {"O": a[("O", 0)], "O-raw": a[("O-raw", 0)], "E_all": a[("E", -1)], "PRIOR_all": a[("PRIOR", -1)],
               "POOL_all": a[("POOL", -1)], "kstar_O": kstar(ks, e, a[("O", 0)]),
               "kstar_Oraw": kstar(ks, e, a[("O-raw", 0)])}
        for k in ks:
            out[f"E_{k}"], out[f"PRIOR_{k}"], out[f"POOL_{k}"] = a[("E", k)], a[("PRIOR", k)], a[("POOL", k)]
        return out

    point = stats(np.ones(len(labs)))
    boots = pd.DataFrame([stats(np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs)))
                          for _ in range(B)])
    d = dict(split=split, n_sig_pairs=int(C.shape[1]), n_articles=len(labs))
    for key in ["O", "O-raw", "E_all", "PRIOR_all", "POOL_all", "kstar_O", "kstar_Oraw"]:
        # inverted-CDF quantiles: k* is inf when E never reaches the target within the budgets tried
        q = np.quantile(boots[key].to_numpy(), [.025, .975], method="inverted_cdf")
        d[key], d[key + "_lo"], d[key + "_hi"] = point[key], q[0], q[1]
    for a_, b_ in [("PRIOR_all", "E_all"), ("O", "O-raw"), ("PRIOR_all", "O"), ("E_all", "O")]:
        diff = boots[a_] - boots[b_]
        d[f"{a_}-{b_}"], d[f"{a_}-{b_}_lo"], d[f"{a_}-{b_}_hi"] = point[a_] - point[b_], diff.quantile(.025), diff.quantile(.975)
    worth.append(d)
    for k in ks:
        for m in ["E", "PRIOR", "POOL"]:
            curves.append(dict(split=split, method=m, k=k, acc=point[f"{m}_{k}"], lo=boots[f"{m}_{k}"].quantile(.025),
                               hi=boots[f"{m}_{k}"].quantile(.975),
                               minus_E=point[f"{m}_{k}"] - point[f"E_{k}"],
                               minus_E_lo=(boots[f"{m}_{k}"] - boots[f"E_{k}"]).quantile(.025),
                               minus_E_hi=(boots[f"{m}_{k}"] - boots[f"E_{k}"]).quantile(.975)))
    print(pd.Series(d).to_string(), flush=True)

pd.DataFrame(worth).round(4).to_csv(TAB / "u4_archive_worth.csv", index=False)
pd.DataFrame(curves).round(4).to_csv(TAB / "u4_curves.csv", index=False)
