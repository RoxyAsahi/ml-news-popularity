"""Main results table (review item R2).

Hyper-parameters are chosen on the validation block (oldest 70% train -> next 10% validation),
then each model is refit on train+validation and evaluated on the test block (newest 20%).
The reference ("best") model in each feature set is the one with the highest VALIDATION AUC.
Test numbers carry 95% intervals from a weekly block bootstrap (whole publication weeks are
resampled, which keeps within-week correlation; article-level intervals are kept for comparison);
model gaps are paired block bootstraps on the same test articles.
HistGB uses temporal early stopping: the number of boosting iterations is picked on the
validation block from the staged predictions (no random internal validation split).

Run from the repo root:  python scripts/main_results.py
"""
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import bootstrap_ci, paired_bootstrap
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
TAB = ROOT / "results" / "tables"

GRIDS = {
    "majority": [{}],
    "logreg": [{"penalty": p, "C": c} for p, c in itertools.product(["l1", "l2"], [0.001, 0.01, 0.1, 1.0])],
    "knn": [{"k": k} for k in [50, 150, 300, 500, 1000, 2000]],
    "rf": [{"min_samples_leaf": l, "max_features": f} for l, f in itertools.product([1, 5, 20], ["sqrt", 0.3])],
    "hgb": [{"learning_rate": lr, "max_leaf_nodes": m, "l2_regularization": l2, "max_iter": 1000}
            for lr, m, l2 in itertools.product([0.03, 0.1], [15, 31, 63], [0.0, 1.0])],
}


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    y = make_label(df.shares, 1400)
    week = df.publish_date.iloc[te].dt.to_period("W").astype(str).to_numpy()
    rows, scores = [], {}
    for fs in ["all", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        for m, grid in GRIDS.items():
            best, best_val = None, -1
            for params in grid:
                mod = fit(get_model(m, 0, **params), X[tr], y[tr])
                if m == "hgb":       # temporal early stopping: best iteration on the validation block
                    for it, pv in enumerate(mod.staged_predict_proba(X[va]), start=1):
                        if it % 10:
                            continue
                        v = roc_auc_score(y[va], pv[:, 1])
                        if v > best_val:
                            best, best_val = {**params, "max_iter": it}, v
                    continue
                v = roc_auc_score(y[va], mod.predict_proba(X[va])[:, 1])
                if v > best_val:
                    best, best_val = params, v
            p = fit(get_model(m, 0, **best), X[trv], y[trv]).predict_proba(X[te])[:, 1]
            scores[(fs, m)] = p
            auc, lo, hi = bootstrap_ci(y[te], p, groups=week)
            _, alo, ahi = bootstrap_ci(y[te], p)                      # article-level, for comparison
            ap, aplo, aphi = bootstrap_ci(y[te], p, metric=average_precision_score, groups=week)
            rows.append({"features": fs, "model": m, "params": json.dumps(best), "val_auc": best_val,
                         "test_auc": auc, "auc_lo": lo, "auc_hi": hi,
                         "auc_lo_article": alo, "auc_hi_article": ahi,
                         "test_pr_auc": ap, "pr_lo": aplo, "pr_hi": aphi})
            print(f"{fs:9s} {m:8s} val {best_val:.3f}  test {auc:.3f} [{lo:.3f}, {hi:.3f}]  {best}")
    out = pd.DataFrame(rows)
    for fs in ["all", "no_kw_ref"]:
        sub = out[(out.features == fs) & (out.model != "majority")]
        best_m = sub.loc[sub.val_auc.idxmax(), "model"]       # chosen on validation, not test
        for i in sub.index:
            m = out.loc[i, "model"]
            if m == best_m:
                continue
            pb = paired_bootstrap(y[te], scores[(fs, best_m)], scores[(fs, m)], groups=week)
            out.loc[i, "vs_best"] = best_m
            out.loc[i, "gap_to_best"] = pb["diff"]
            out.loc[i, "gap_lo"], out.loc[i, "gap_hi"] = pb["lo"], pb["hi"]
    # same model, with vs without share-derived features
    for m in GRIDS:
        if m == "majority":
            continue
        pb = paired_bootstrap(y[te], scores[("all", m)], scores[("no_kw_ref", m)], groups=week)
        i = out[(out.features == "no_kw_ref") & (out.model == m)].index[0]
        out.loc[i, "drop_vs_all"], out.loc[i, "drop_lo"], out.loc[i, "drop_hi"] = pb["diff"], pb["lo"], pb["hi"]
    out.to_csv(TAB / "main_results.csv", index=False)
    print(out.drop(columns=["params"]).round(3).to_string())


if __name__ == "__main__":
    main()
