"""E5 (RQ3) - How much training data is needed?

Random subsamples (5 seeds) of the chronological training block (oldest 80%),
evaluated on the newest 20%. Also reports training AUC to show the generalisation gap.

Run from the repo root:  python scripts/e5_learning_curve.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FRACS = [0.005, 0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["logreg", "knn", "rf", "hgb"]
NAMES = {"logreg": "Logistic regression", "knn": "kNN", "rf": "Random forest", "hgb": "HistGB"}


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    y = make_label(df.shares, 1400)
    rows = []
    for fs in ["all", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        for frac in FRACS:
            k = int(len(trv) * frac)
            for seed in (SEEDS if frac < 1 else [0]):
                idx = np.sort(np.random.RandomState(seed).choice(trv, k, replace=False))
                for m in MODELS:
                    # kNN: k scales with data so that k never exceeds 10% of the sample
                    mod = fit(get_model(m, seed, k=max(5, min(50, k // 10)), n_estimators=200), X[idx], y[idx])
                    rows.append({"features": fs, "frac": frac, "n_train": k, "seed": seed, "model": m,
                                 "train_auc": roc_auc_score(y[idx], mod.predict_proba(X[idx])[:, 1]),
                                 "test_auc": roc_auc_score(y[te], mod.predict_proba(X[te])[:, 1])})
            print(fs, frac, "done")
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "e5_learning_curve.csv", index=False)
    summ = out.groupby(["features", "model", "n_train"])[["train_auc", "test_auc"]].agg(["mean", "std"]).round(3)
    summ.to_csv(TAB / "e5_learning_curve_summary.csv")
    print(summ.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    colors = {"logreg": "0.55", "knn": "C2", "rf": "C1", "hgb": "C0"}
    for ax, fs, title in zip(axes, ["all", "no_kw_ref"], ["all features", "without kw_* / self_reference_*"]):
        for m in MODELS:
            d = out[(out.features == fs) & (out.model == m)].groupby("n_train")["test_auc"].agg(["mean", "std"])
            ax.errorbar(d.index, d["mean"], yerr=d["std"].fillna(0), marker="o", ms=3, capsize=2,
                        color=colors[m], label=NAMES[m])
        ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel("training articles (log scale)")
    axes[0].set_ylabel("test ROC-AUC (newest 20%)"); axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "e5_learning_curve.png", dpi=200)


if __name__ == "__main__":
    main()
