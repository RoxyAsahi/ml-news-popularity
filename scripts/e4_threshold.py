"""E4 (RQ3) - How does the choice of popularity threshold change the task?

Thresholds at the 50/75/90/95th percentile of training-block shares. For each we train
dedicated classifiers and compare with ONE regressor on log(shares) whose predictions
are used as a ranking score for every threshold. Models are fit on the training block;
the F1 decision cutoff is chosen on the validation block; the test block is scored once.
Bootstrap CIs; paired bootstrap for classifier vs regressor.

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
from sklearn.metrics import average_precision_score, roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import bootstrap_ci, classification_report, paired_bootstrap
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
QUANTILES = [0.5, 0.75, 0.9, 0.95]


def f1_cutoff(y_val, s_val):
    """Decision cutoff that maximises F1 on the validation block (never looks at test)."""
    from sklearn.metrics import f1_score
    cands = np.quantile(s_val, np.linspace(0.01, 0.99, 99))
    return float(max(cands, key=lambda c: f1_score(y_val, s_val >= c)))


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    rows, pairs = [], []
    for fs in ["all", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        # everything is fit on the training block; the validation block only picks cutoffs
        reg = HistGradientBoostingRegressor(random_state=0).fit(X[tr], np.log1p(df.shares.iloc[tr]))
        s_va, s_te = reg.predict(X[va]), reg.predict(X[te])
        rho = spearmanr(s_te, df.shares.iloc[te]).statistic
        print(f"{fs}: regressor Spearman(pred, shares) on test = {rho:.3f}")
        for q in QUANTILES:
            thr = df.shares.iloc[tr].quantile(q)
            y = make_label(df.shares, thr)
            test_scores = {}
            for m in ["logreg", "hgb"]:
                mod = fit(get_model(m, 0), X[tr], y[tr])
                p_va, p_te = mod.predict_proba(X[va])[:, 1], mod.predict_proba(X[te])[:, 1]
                test_scores[m] = p_te
                r = classification_report(y[te], p_te, threshold=f1_cutoff(y[va], p_va))
                auc, lo, hi = bootstrap_ci(y[te], p_te)
                rows.append({"features": fs, "quantile": q, "threshold": thr, "model": m, **r,
                             "auc_lo": lo, "auc_hi": hi, "n_pos_test": int(y[te].sum())})
            test_scores["reg"] = s_te
            r = classification_report(y[te], s_te, threshold=f1_cutoff(y[va], s_va))
            auc, lo, hi = bootstrap_ci(y[te], s_te)
            rows.append({"features": fs, "quantile": q, "threshold": thr, "model": "log_regression_score",
                         **r, "auc_lo": lo, "auc_hi": hi, "n_pos_test": int(y[te].sum()), "spearman": rho})
            for metric_name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
                pb = paired_bootstrap(y[te], test_scores["hgb"], test_scores["reg"], metric=metric)
                pairs.append({"features": fs, "quantile": q, "metric": metric_name,
                              "hgb_minus_regressor": pb["diff"], "lo": pb["lo"], "hi": pb["hi"]})
    out = pd.DataFrame(rows)
    out["pr_auc_lift"] = out["pr_auc"] / out["pos_rate"]
    out.to_csv(TAB / "e4_threshold.csv", index=False)
    pd.DataFrame(pairs).to_csv(TAB / "e4_classifier_vs_regressor.csv", index=False)
    print(out[["features", "quantile", "threshold", "model", "n_pos_test", "pos_rate", "roc_auc", "auc_lo",
               "auc_hi", "pr_auc", "pr_auc_lift", "f1"]].round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string())

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
