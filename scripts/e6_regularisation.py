"""E6 (RQ3) - Regularisation strength and generalisation.

1. Logistic regression, L1 and L2, C from 1e-4 to 1e2: train vs validation AUC,
   at full training size and with only 1,000 training articles.
2. L1 path: which features survive the strongest penalties.
3. kNN neighbourhood size k as a smoothness knob (k = 1 ... 1000).
Validation = the 10% block after training (test block untouched).

Run from the repo root:  python scripts/e6_regularisation.py
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
CS = np.logspace(-4, 2, 13)


def main():
    df = load()
    n = len(df)
    tr, va, _ = temporal_split(n, 0.7, 0.1)
    cols = feature_set(df, "all")
    X = df[cols].to_numpy()
    y = make_label(df.shares, 1400)
    small = np.sort(np.random.RandomState(0).choice(tr, 1000, replace=False))

    rows, coefs = [], []
    for size, idx in [("full", tr), ("1000", small)]:
        for pen in ["l2", "l1"]:
            for C in CS:
                mod = fit(get_model("logreg", 0, C=C, penalty=pen), X[idx], y[idx])
                w = mod[-1].coef_.ravel()
                rows.append({"train_size": size, "penalty": pen, "C": C,
                             "train_auc": roc_auc_score(y[idx], mod.predict_proba(X[idx])[:, 1]),
                             "val_auc": roc_auc_score(y[va], mod.predict_proba(X[va])[:, 1]),
                             "n_nonzero": int((np.abs(w) > 1e-8).sum())})
                if size == "full" and pen == "l1":
                    coefs.append(pd.Series(w, index=cols, name=C))
    lr = pd.DataFrame(rows)
    lr.to_csv(TAB / "e6_logreg_C.csv", index=False)
    print(lr.round(3).to_string())

    path = pd.DataFrame(coefs)
    path.to_csv(TAB / "e6_l1_path.csv")
    # order in which features enter the model as C grows
    entry = {c: path.index[(path[c].abs() > 1e-8).to_numpy()].min() for c in cols
             if (path[c].abs() > 1e-8).any()}
    entry = pd.Series(entry, name="first_C_nonzero").sort_values()
    entry.to_csv(TAB / "e6_l1_entry_order.csv")
    print("first features to enter the L1 model:\n", entry.head(12))

    krows = []
    for k in [1, 5, 15, 50, 150, 500, 1000]:
        mod = fit(get_model("knn", 0, k=k), X[tr], y[tr])
        krows.append({"k": k, "val_auc": roc_auc_score(y[va], mod.predict_proba(X[va])[:, 1])})
    kn = pd.DataFrame(krows)
    kn.to_csv(TAB / "e6_knn_k.csv", index=False)
    print(kn.round(3).to_string())

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    for ax, size, title in [(axes[0], "full", "LogReg, 28k training articles"),
                            (axes[1], "1000", "LogReg, 1,000 training articles")]:
        for pen, c in [("l2", "0.5"), ("l1", "C0")]:
            d = lr[(lr.train_size == size) & (lr.penalty == pen)]
            ax.plot(d["C"], d["val_auc"], marker="o", ms=3, color=c, label=f"{pen.upper()} validation")
            ax.plot(d["C"], d["train_auc"], ls="--", color=c, label=f"{pen.upper()} train")
        ax.set_xscale("log"); ax.set_xlabel("C (smaller = stronger penalty)"); ax.set_title(title, fontsize=9)
        ax.set_ylim(0.45, 0.95)
    axes[0].set_ylabel("ROC-AUC"); axes[0].legend(fontsize=7)
    axes[2].plot(kn["k"], kn["val_auc"], marker="o", ms=3, color="C2")
    axes[2].set_xscale("log"); axes[2].set_xlabel("k neighbours (larger = smoother)")
    axes[2].set_title("kNN, validation ROC-AUC", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "e6_regularisation.png", dpi=200)


if __name__ == "__main__":
    main()
