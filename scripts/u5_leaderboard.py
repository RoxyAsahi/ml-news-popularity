"""U5: does the usual (observational) validation metric pick the model that ranks headlines best?

For every model of u2 (representation x training signal): the observational metric (Spearman correlation of
the prediction with logit CTR across held-out tests, fold mean) and the interventional metric (accuracy on
significantly different within-test pairs, with article-cluster bootstrap CIs from the per-pair results), plus
bootstrap CIs for key differences and the rank correlation between the two metrics across models.
Usage: python scripts/u5_leaderboard.py [split ...]
Output: results/tables/u5_leaderboard.csv, results/tables/u5_differences.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from newspop.data import ROOT

TAB = ROOT / "results" / "tables"
splits = sys.argv[1:] or ["exploratory", "confirmatory", "holdout"]
B = 1000
PAIRS = [(("GloVe+H15", "OBS-FE"), ("GloVe+H15", "OBS")), (("GloVe+H15", "OBS"), ("GloVe+H15", "OBS-HGB")),
         (("GloVe+H15", "ORACLE"), ("GloVe+H15", "OBS-FE")), (("GloVe+H15", "NAT"), ("GloVe+H15", "OBS-FE")),
         (("GloVe+H15", "MIX (OBS-FE + NAT)"), ("GloVe+H15", "OBS-FE")), (("GloVe+H15", "OBS"), ("H15", "OBS"))]
board, diffs = [], []
for split in splits:
    f = np.load(TAB / f"u2_pairs_{split}.npz")
    keys = [tuple(c.split("|")) for c in f["configs"]]
    C, art = f["correct"], f["article"]
    labs, inv = np.unique(art, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=c, minlength=len(labs)) for c in C])
    cnt = np.bincount(inv, minlength=len(labs))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs)) for _ in range(B)])
    acc = sums.sum(1) / cnt.sum()
    accb = (sums @ W.T) / (cnt @ W.T)                       # configs x B
    R = pd.read_csv(TAB / f"u2_ranking_{split}.csv").groupby(["rep", "train"]).across_spearman.mean()
    for i, k in enumerate(keys):
        board.append(dict(split=split, rep=k[0], train=k[1], across_spearman=R[k], within_acc_sig=acc[i],
                          lo=np.quantile(accb[i], .025), hi=np.quantile(accb[i], .975)))
    for a, b in PAIRS:
        if a in keys and b in keys:
            ia, ib = keys.index(a), keys.index(b)
            dd = accb[ia] - accb[ib]
            diffs.append(dict(split=split, a=" / ".join(a), b=" / ".join(b), diff=acc[ia] - acc[ib],
                              lo=np.quantile(dd, .025), hi=np.quantile(dd, .975)))
    sub = pd.DataFrame(board)
    sub = sub[sub.split == split]
    rho = spearmanr(sub.across_spearman, sub.within_acc_sig)[0]
    g = sub[sub.rep == "GloVe+H15"]
    rho_g = spearmanr(g.across_spearman, g.within_acc_sig)[0]
    diffs.append(dict(split=split, a="rank corr. across vs within, all models", b="", diff=rho))
    diffs.append(dict(split=split, a="rank corr. across vs within, GloVe models", b="", diff=rho_g))
B_ = pd.DataFrame(board).round(4)
D_ = pd.DataFrame(diffs).round(4)
B_.to_csv(TAB / "u5_leaderboard.csv", index=False)
D_.to_csv(TAB / "u5_differences.csv", index=False)
print(B_.to_string(index=False))
print(D_.to_string(index=False))
