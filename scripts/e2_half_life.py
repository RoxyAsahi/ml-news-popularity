"""E2 (RQ1) - Does down-weighting older training articles help on newer ones?

Half-life sample weights 0.5 ** (age / half_life), age in days measured from the newest
training article (idea inspired by the TimeDecay mechanism in VCPToolBox; independent
implementation). Evaluated on 5 rolling-origin folds; for each half-life we report the paired
per-fold AUC difference against unweighted training (95% t-interval).

Run from the repo root:  python scripts/e2_half_life.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import paired_fold_diff
from newspop.models import fit, get_model
from newspop.splits import recency_weights, rolling_origin

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
HALF_LIVES = [None, 30, 90, 180, 365, 730]


def main():
    df = load()
    n = len(df)
    y = make_label(df.shares, 1400)
    X = df[feature_set(df, "all")].to_numpy()
    folds = list(rolling_origin(n, 5, 0.4))
    aucs = {}
    for m in ["logreg", "hgb"]:
        for hl in HALF_LIVES:
            per_fold = []
            for a, b in folds:
                age = (df.publish_date.iloc[a].max() - df.publish_date.iloc[a]).dt.days.to_numpy()
                mod = fit(get_model(m, 0), X[a], y[a], sample_weight=recency_weights(age, hl))
                per_fold.append(roc_auc_score(y[b], mod.predict_proba(X[b])[:, 1]))
            aucs[(m, hl)] = per_fold
            print(m, hl, np.round(per_fold, 3))
    rows = []
    for (m, hl), v in aucs.items():
        row = {"model": m, "half_life_days": "none" if hl is None else hl, "mean_auc": np.mean(v)}
        if hl is not None:
            d = paired_fold_diff(v, aucs[(m, None)])
            row.update({"diff_vs_unweighted": d["diff"], "lo": d["lo"], "hi": d["hi"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "e2_half_life.csv", index=False)
    print(out.round(4).to_string())

    fig, ax = plt.subplots(figsize=(5.5, 3))
    for m, c, off in [("hgb", "C0", 0.03), ("logreg", "0.5", -0.03)]:
        d = out[(out.model == m) & (out.half_life_days != "none")]
        x = np.log10(d.half_life_days.astype(float)) + off
        ax.errorbar(x, d.diff_vs_unweighted, yerr=[d.diff_vs_unweighted - d.lo, d.hi - d.diff_vs_unweighted],
                    marker="o", ms=3, capsize=2, color=c,
                    label={"hgb": "gradient boosting", "logreg": "logistic regression"}[m])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(np.log10(HALF_LIVES[1:]), [str(h) for h in HALF_LIVES[1:]])
    ax.set_xlabel("half-life of sample weights (days)")
    ax.set_ylabel("ROC-AUC change vs unweighted\n(paired, 5 rolling folds)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "e2_half_life.png", dpi=200)


if __name__ == "__main__":
    main()
