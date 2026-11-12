"""Pilot run for the plan: split type x feature set x model, threshold sweep,
learning curve and half-life weighting. Untuned; numbers only guide the plan.

Run from the repo root:  python scripts/01_pilot.py
"""
import warnings

import numpy as np
import pandas as pd

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import classification_report
from newspop.models import fit, get_model
from newspop.splits import random_split, recency_weights

warnings.filterwarnings("ignore")
TAB = ROOT / "results" / "tables"
SEED = 0


def main():
    df = load()
    n = len(df)
    cut = int(n * 0.8)
    splits = {"random": random_split(n, 0.2, SEED), "temporal": (np.arange(cut), np.arange(cut, n))}
    rows = []

    # A. split x feature set x model (median threshold fixed at 1400, as in Fernandes et al.)
    y = make_label(df.shares, 1400)
    for fs in ["all", "no_kw", "no_kw_ref"]:
        X = df[feature_set(df, fs)].to_numpy()
        for split, (tr, te) in splits.items():
            for m in ["majority", "logreg", "knn", "hgb"]:
                mod = fit(get_model(m, SEED), X[tr], y[tr])
                r = classification_report(y[te], mod.predict_proba(X[te])[:, 1])
                rows.append({"exp": "split_features", "features": fs, "split": split, "model": m, **r})
                print("A", fs, split, m, round(r["roc_auc"], 3))

    X = df[feature_set(df, "all")].to_numpy()
    tr, te = splits["temporal"]

    # B. threshold sweep (quantiles computed on the training part only)
    for q in [0.5, 0.75, 0.9, 0.95]:
        thr = df.shares.iloc[tr].quantile(q)
        yq = make_label(df.shares, thr)
        mod = fit(get_model("hgb", SEED), X[tr], yq[tr])
        r = classification_report(yq[te], mod.predict_proba(X[te])[:, 1])
        rows.append({"exp": "threshold", "quantile": q, "threshold": thr, "model": "hgb", **r})
        print("B", q, thr, round(r["roc_auc"], 3), round(r["pr_auc"], 3))

    # C. learning curve (most recent k training articles)
    for frac in [0.01, 0.03, 0.1, 0.3, 1.0]:
        idx = tr[-int(len(tr) * frac):]
        for m in ["logreg", "knn", "hgb"]:
            mod = fit(get_model(m, SEED, k=min(50, len(idx) // 10)), X[idx], y[idx])
            r = classification_report(y[te], mod.predict_proba(X[te])[:, 1])
            rows.append({"exp": "learning_curve", "frac": frac, "model": m, **r})
            print("C", frac, m, round(r["roc_auc"], 3))

    # D. half-life recency weighting
    age = (df.publish_date.iloc[tr].max() - df.publish_date.iloc[tr]).dt.days.to_numpy()
    for hl in [None, 30, 90, 180, 365]:
        mod = fit(get_model("hgb", SEED), X[tr], y[tr], sample_weight=recency_weights(age, hl))
        r = classification_report(y[te], mod.predict_proba(X[te])[:, 1])
        rows.append({"exp": "half_life", "half_life": hl, "model": "hgb", **r})
        print("D", hl, round(r["roc_auc"], 3))

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "pilot_results.csv", index=False)
    print(f"wrote {TAB / 'pilot_results.csv'}")


if __name__ == "__main__":
    main()
