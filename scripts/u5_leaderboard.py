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

# ---- observational metrics with paired bootstrap (needs u2_heldout_<split>.csv.gz): the usual held-out
# Spearman with logit CTR, and the same with CTR demeaned within publication week ("validate on the estimand
# you train for"); per model, and gaps between same-representation models, resampling articles.
from newspop import upw_rank as ur  # noqa: E402

obs_rows = []
for split in splits:
    path = TAB / f"u2_heldout_{split}.csv.gz"
    if not path.exists():
        continue
    H = pd.read_csv(path)
    art = ur.setup(split).article
    H["article"] = art[H.test]
    H["y_wk"] = H.y - H.groupby(["rep", "train", "week"]).y.transform("mean")
    H["p_wk"] = H.pred - H.groupby(["rep", "train", "week"]).pred.transform("mean")
    g = H[H.rep == "GloVe+H15"]
    models = sorted(g.train.unique())
    P = g.pivot_table(index="test", columns="train", values=["pred", "y", "y_wk", "p_wk"])
    arts = art[P.index.to_numpy()]
    labs, inv = np.unique(arts, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(labs))]
    rng = np.random.default_rng(0)

    def metrics(ix):
        out = {}
        for m in models:
            out[("raw", m)] = spearmanr(P[("pred", m)].to_numpy()[ix], P[("y", m)].to_numpy()[ix])[0]
            out[("week", m)] = spearmanr(P[("p_wk", m)].to_numpy()[ix], P[("y_wk", m)].to_numpy()[ix])[0]
        return out

    point = metrics(np.arange(len(P)))
    boots = [metrics(np.concatenate([members[c] for c in rng.integers(0, len(labs), len(labs))])) for _ in range(300)]
    acc = pd.DataFrame(board).query("split == @split and rep == 'GloVe+H15'").set_index("train").within_acc_sig
    for kind in ["raw", "week"]:
        for m in models:
            bb = np.array([b[(kind, m)] for b in boots])
            dd = np.array([b[(kind, m)] - b[(kind, "OBS")] for b in boots])
            obs_rows.append(dict(split=split, metric=kind, train=m, spearman=point[(kind, m)],
                                 lo=np.quantile(bb, .025), hi=np.quantile(bb, .975),
                                 minus_OBS=point[(kind, m)] - point[(kind, "OBS")],
                                 minus_OBS_lo=np.quantile(dd, .025), minus_OBS_hi=np.quantile(dd, .975),
                                 within_acc_sig=acc[m]))
        sub = [r for r in obs_rows if r["split"] == split and r["metric"] == kind]
        rho = spearmanr([r["spearman"] for r in sub], [r["within_acc_sig"] for r in sub])[0]
        obs_rows.append(dict(split=split, metric=kind, train="rank corr. with decision accuracy", spearman=rho))
if obs_rows:
    O_ = pd.DataFrame(obs_rows).round(4)
    O_.to_csv(TAB / "u5_observational_metrics.csv", index=False)
    print(O_.to_string(index=False))
