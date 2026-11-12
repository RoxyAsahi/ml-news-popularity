"""E1 (RQ1) - Is a model trained on older articles still good on newer ones?

1. Random vs chronological split (70/10/20), several models; random split repeated over seeds.
2. Rolling-origin evaluation (5 expanding-window folds).
3. Month-by-month evaluation after the training period: positive rate and ROC-AUC.
4. Adversarial validation: how well can a classifier tell old from new articles
   from the features alone? (covariate shift)

Run from the repo root:  python scripts/e1_drift.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import classification_report
from newspop.models import fit, get_model
from newspop.splits import random_split, rolling_origin, temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
THRESHOLD = 1400          # Fernandes et al. (2015); equals the overall median
MODELS = ["majority", "logreg", "knn", "rf", "hgb"]
SEEDS = [0, 1, 2, 3, 4]


def split_comparison(X, y, n):
    """Random vs chronological; same 80/20 train+val / test proportions."""
    rows = []
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])               # no tuning in E1 -> use train+val
    for m in MODELS:
        mod = fit(get_model(m, 0), X[trv], y[trv])
        rows.append({"split": "temporal", "seed": 0, "model": m,
                     **classification_report(y[te], mod.predict_proba(X[te])[:, 1])})
        for s in SEEDS:
            a, b = random_split(n, 0.2, s)
            mod = fit(get_model(m, s), X[a], y[a])
            rows.append({"split": "random", "seed": s, "model": m,
                         **classification_report(y[b], mod.predict_proba(X[b])[:, 1])})
        print("split comparison done:", m)
    return pd.DataFrame(rows)


def rolling(X, y, n):
    rows = []
    for k, (tr, te) in enumerate(rolling_origin(n, n_folds=5, min_train=0.4)):
        for m in MODELS[1:]:
            mod = fit(get_model(m, 0), X[tr], y[tr])
            rows.append({"fold": k, "n_train": len(tr), "model": m,
                         **classification_report(y[te], mod.predict_proba(X[te])[:, 1])})
    return pd.DataFrame(rows)


def monthly(df, X, y, n):
    """Train once on the oldest 70%, then score every later month separately."""
    tr, _, _ = temporal_split(n, 0.7, 0.1)
    after = np.arange(len(tr), n)
    month = df["publish_date"].dt.to_period("M").astype(str).to_numpy()
    rows = []
    for m in ["logreg", "hgb"]:
        mod = fit(get_model(m, 0), X[tr], y[tr])
        p = mod.predict_proba(X[after])[:, 1]
        for mo in pd.unique(month[after]):
            sel = month[after] == mo
            if sel.sum() < 200:      # skip a partial month at the split boundary
                continue
            rows.append({"month": mo, "model": m, "n": int(sel.sum()),
                         "pos_rate": y[after][sel].mean(),
                         "pred_pos_rate": (p[sel] >= 0.5).mean(),
                         "roc_auc": roc_auc_score(y[after][sel], p[sel])})
    base = (pd.DataFrame({"month": month, "y": y}).groupby("month")["y"]
            .agg(pos_rate="mean", n="size").reset_index())
    return pd.DataFrame(rows), base, month[tr[-1]]


def adversarial(X, n):
    """Label oldest 70% as 0 and newest 30% as 1; 5-fold CV AUC of telling them apart."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    z = (np.arange(n) >= int(n * 0.7)).astype(int)
    p = cross_val_predict(get_model("hgb", 0), X, z, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    return roc_auc_score(z, p)


def main():
    df = load()
    n = len(df)
    y = make_label(df.shares, THRESHOLD)
    cols = feature_set(df, "all")
    X = df[cols].to_numpy()

    comp = split_comparison(X, y, n)
    comp.to_csv(TAB / "e1_split_comparison.csv", index=False)
    summary = (comp.groupby(["model", "split"])[["roc_auc", "pr_auc", "f1", "balanced_acc"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(TAB / "e1_split_summary.csv")
    print(summary)

    roll = rolling(X, y, n)
    roll.to_csv(TAB / "e1_rolling.csv", index=False)
    print(roll.groupby("model")[["roc_auc", "balanced_acc"]].agg(["mean", "std"]).round(3))

    mon, base, last_train_month = monthly(df, X, y, n)
    mon.to_csv(TAB / "e1_monthly_eval.csv", index=False)

    adv = {}
    for fs in ["all", "no_kw_ref"]:
        adv[fs] = adversarial(df[feature_set(df, fs)].to_numpy(), n)
    pd.Series(adv, name="adversarial_auc").to_csv(TAB / "e1_adversarial.csv")
    print("adversarial AUC (old vs new):", {k: round(v, 3) for k, v in adv.items()})

    # ---- figure: base rate over time + monthly AUC after training period
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    x = pd.PeriodIndex(base["month"], freq="M").to_timestamp()
    ax1.plot(x, base["pos_rate"], marker="o", ms=3, color="0.35")
    ax1.axhline(y.mean(), color="0.6", ls=":", lw=1)
    cut = pd.Period(last_train_month, freq="M").to_timestamp()
    for ax in (ax1, ax2):
        ax.axvline(cut, color="C3", ls="--", lw=1)
    ax1.text(cut, ax1.get_ylim()[1], " end of training data", color="C3", va="top", fontsize=8)
    ax1.set_ylabel("share popular\n(>= 1400 shares)")
    for m, c in [("hgb", "C0"), ("logreg", "0.5")]:
        s = mon[mon.model == m]
        ax2.plot(pd.PeriodIndex(s["month"], freq="M").to_timestamp(), s["roc_auc"],
                 marker="o", ms=3, color=c, label={"hgb": "gradient boosting", "logreg": "logistic regression"}[m])
    ax2.set_ylabel("ROC-AUC per month"); ax2.legend(fontsize=8, loc="lower left")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIG / "e1_drift_monthly.png", dpi=200)

    # ---- figure: random vs temporal
    agg = comp[comp.model != "majority"].groupby(["model", "split"])["roc_auc"].agg(["mean", "std"]).reset_index()
    order = ["logreg", "knn", "rf", "hgb"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    w = 0.38
    for j, (sp, c) in enumerate([("random", "0.65"), ("temporal", "C0")]):
        s = agg[agg.split == sp].set_index("model").loc[order]
        ax.bar(np.arange(4) + (j - 0.5) * w, s["mean"], w, yerr=s["std"].fillna(0),
               color=c, label=sp, capsize=2)
    ax.set_xticks(np.arange(4), ["LogReg", "kNN", "RF", "HistGB"])
    ax.set_ylim(0.6, 0.76); ax.set_ylabel("ROC-AUC (test)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "e1_split_comparison.png", dpi=200)
    print("done")


if __name__ == "__main__":
    main()
