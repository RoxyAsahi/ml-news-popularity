"""Main results table (review item R2).

Hyper-parameters are chosen on the validation block (oldest 70% train -> next 10% validation),
then each model is refit on train+validation and evaluated ONCE on the test block (newest 20%).
All test numbers carry 95% bootstrap intervals; every model is compared with the best one by a
paired bootstrap on the same test articles.

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
    "knn": [{"k": k} for k in [50, 150, 300, 500]],
    "rf": [{"min_samples_leaf": l, "max_features": f} for l, f in itertools.product([1, 5, 20], ["sqrt", 0.3])],
    "hgb": [{"learning_rate": lr, "max_leaf_nodes": m, "l2_regularization": l2, "max_iter": it}
            for lr, m, l2, it in [(0.1, 31, 0.0, 100), (0.05, 31, 0.0, 300), (0.05, 15, 1.0, 300),
                                  (0.03, 15, 1.0, 500), (0.05, 63, 1.0, 200), (0.1, 15, 0.0, 100)]],
}


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    y = make_label(df.shares, 1400)
    rows, scores = [], {}
    for fs in ["all", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        for m, grid in GRIDS.items():
            best, best_val = None, -1
            for params in grid:
                mod = fit(get_model(m, 0, **params), X[tr], y[tr])
                v = roc_auc_score(y[va], mod.predict_proba(X[va])[:, 1])
                if v > best_val:
                    best, best_val = params, v
            p = fit(get_model(m, 0, **best), X[trv], y[trv]).predict_proba(X[te])[:, 1]
            scores[(fs, m)] = p
            auc, lo, hi = bootstrap_ci(y[te], p)
            ap, aplo, aphi = bootstrap_ci(y[te], p, metric=average_precision_score)
            rows.append({"features": fs, "model": m, "params": json.dumps(best), "val_auc": best_val,
                         "test_auc": auc, "auc_lo": lo, "auc_hi": hi,
                         "test_pr_auc": ap, "pr_lo": aplo, "pr_hi": aphi})
            print(f"{fs:9s} {m:8s} val {best_val:.3f}  test {auc:.3f} [{lo:.3f}, {hi:.3f}]  {best}")
    out = pd.DataFrame(rows)
    for fs in ["all", "no_kw_ref"]:
        sub = out[(out.features == fs) & (out.model != "majority")]
        best_m = sub.loc[sub.test_auc.idxmax(), "model"]
        for i in sub.index:
            m = out.loc[i, "model"]
            if m == best_m:
                continue
            pb = paired_bootstrap(y[te], scores[(fs, best_m)], scores[(fs, m)])
            out.loc[i, "vs_best"] = best_m
            out.loc[i, "gap_to_best"] = pb["diff"]
            out.loc[i, "gap_lo"], out.loc[i, "gap_hi"] = pb["lo"], pb["hi"]
    # same model, with vs without share-derived features
    for m in GRIDS:
        if m == "majority":
            continue
        pb = paired_bootstrap(y[te], scores[("all", m)], scores[("no_kw_ref", m)])
        i = out[(out.features == "no_kw_ref") & (out.model == m)].index[0]
        out.loc[i, "drop_vs_all"], out.loc[i, "drop_lo"], out.loc[i, "drop_hi"] = pb["diff"], pb["lo"], pb["hi"]
    out.to_csv(TAB / "main_results.csv", index=False)
    print(out.drop(columns=["params"]).round(3).to_string())


if __name__ == "__main__":
    main()
