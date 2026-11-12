"""E4 (RQ3) - How does the choice of popularity threshold change the task?

Thresholds at the 50/75/90/95th percentile of training shares. For each we train
dedicated classifiers and compare with ONE regressor on log(shares) whose predictions
are used as a ranking score for every threshold.

Run from the repo root:  python scripts/e4_threshold.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import classification_report
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
QUANTILES = [0.5, 0.75, 0.9, 0.95]


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    rows = []
    for fs in ["all", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        reg = HistGradientBoostingRegressor(random_state=0).fit(X[trv], np.log1p(df.shares.iloc[trv]))
        score = reg.predict(X[te])
        rho = spearmanr(score, df.shares.iloc[te]).statistic
        print(f"{fs}: regressor Spearman(pred, shares) on test = {rho:.3f}")
        for q in QUANTILES:
            thr = df.shares.iloc[trv].quantile(q)
            y = make_label(df.shares, thr)
            for m in ["logreg", "hgb"]:
                mod = fit(get_model(m, 0), X[trv], y[trv])
                p = mod.predict_proba(X[te])[:, 1]
                # decision threshold = training positive rate quantile, so the
                # classifier flags as many articles as are popular in training
                cut = np.quantile(p, 1 - y[trv].mean())
                r = classification_report(y[te], p, threshold=cut)
                rows.append({"features": fs, "quantile": q, "threshold": thr, "model": m, **r})
            r = classification_report(y[te], score, threshold=np.quantile(score, 1 - y[trv].mean()))
            rows.append({"features": fs, "quantile": q, "threshold": thr, "model": "log_regression_score",
                         **r, "spearman": rho})
    out = pd.DataFrame(rows)
    out["pr_auc_lift"] = out["pr_auc"] / out["pos_rate"]
    out.to_csv(TAB / "e4_threshold.csv", index=False)
    print(out[["features", "quantile", "threshold", "model", "pos_rate", "roc_auc", "pr_auc",
               "pr_auc_lift", "f1"]].round(3).to_string())

    # figure: ROC-AUC flat, PR-AUC falls, lift rises (HistGB, all features)
    s = out[(out.features == "all")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    names = {"hgb": "HistGB classifier", "logreg": "LogReg classifier",
             "log_regression_score": "one log-shares regressor"}
    styles = {"hgb": ("C0", "o"), "logreg": ("0.55", "s"), "log_regression_score": ("0.2", "^")}
    for m in ["hgb", "logreg", "log_regression_score"]:
        d = s[s.model == m]
        c, mk = styles[m]
        for ax, col in zip(axes, ["roc_auc", "pr_auc", "pr_auc_lift"]):
            ax.plot(d["quantile"] * 100, d[col], marker=mk, color=c, label=names[m])
    pr = s[s.model == "hgb"]
    axes[1].plot(pr["quantile"] * 100, pr["pos_rate"], ls=":", color="0.6", label="chance (positive rate)")
    for ax, t in zip(axes, ["ROC-AUC", "PR-AUC", "PR-AUC / positive rate"]):
        ax.set_title(t, fontsize=9); ax.set_xlabel("threshold percentile"); ax.set_xticks([50, 75, 90, 95])
    axes[0].set_ylim(0.6, 0.8)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "e4_threshold.png", dpi=200)


if __name__ == "__main__":
    main()
