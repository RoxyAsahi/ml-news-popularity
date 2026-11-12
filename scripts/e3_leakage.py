"""E3 (RQ2) - How much does performance depend on features derived from earlier
articles' share counts (kw_*, self_reference_*)? These are point-in-time, not leakage;
see scripts/e3b_point_in_time.py.

1. Feature-set ablation under the chronological split and rolling-origin folds.
2. Drop-one-group ablation for every feature group.
3. Grouped permutation importance on the validation block.
4. Time signal per feature: Spearman correlation with the publication date.

Run from the repo root:  python scripts/e3_leakage.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from newspop.data import ROOT, feature_groups, feature_set, load, make_label
from newspop.evaluate import classification_report
from newspop.models import fit, get_model
from newspop.splits import rolling_origin, temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
THRESHOLD = 1400
MODELS = ["logreg", "hgb"]


def feature_sets(df):
    g = feature_groups(df)
    every = feature_set(df, "all")
    share_derived = g["keyword"] + g["self_reference"]
    sets = {
        "all": every,
        "no_kw": feature_set(df, "no_kw"),
        "no_self_ref": [c for c in every if c not in g["self_reference"]],
        "no_kw_ref": feature_set(df, "no_kw_ref"),
        "only_kw_ref": share_derived,
    }
    for name, cols in g.items():
        if name in ("keyword", "self_reference"):
            continue
        sets[f"drop_{name}"] = [c for c in every if c not in cols]
    return sets


def ablation(df, y, sets):
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    rows = []
    for name, cols in sets.items():
        X = df[cols].to_numpy()
        for m in MODELS:
            mod = fit(get_model(m, 0), X[trv], y[trv])
            r = classification_report(y[te], mod.predict_proba(X[te])[:, 1])
            roll = []
            for a, b in rolling_origin(n, 5, 0.4):
                roll.append(roc_auc_score(y[b], fit(get_model(m, 0), X[a], y[a]).predict_proba(X[b])[:, 1]))
            rows.append({"feature_set": name, "n_features": len(cols), "model": m, **r,
                         "rolling_auc_mean": np.mean(roll), "rolling_auc_std": np.std(roll)})
            print(f"{name:18s} {m:6s} test AUC {r['roc_auc']:.3f}  rolling {np.mean(roll):.3f}±{np.std(roll):.3f}")
    return pd.DataFrame(rows)


def grouped_permutation(df, y, n_repeats=10, seed=0):
    """Permute all columns of a group together on the validation block (test stays untouched)."""
    n = len(df)
    tr, va, _ = temporal_split(n, 0.7, 0.1)
    cols = feature_set(df, "all")
    X = df[cols].to_numpy()
    groups = feature_groups(df)
    rng = np.random.RandomState(seed)
    rows = []
    for m in MODELS:
        mod = fit(get_model(m, 0), X[tr], y[tr])
        base = roc_auc_score(y[va], mod.predict_proba(X[va])[:, 1])
        for gname, gcols in groups.items():
            idx = [cols.index(c) for c in gcols]
            drops = []
            for _ in range(n_repeats):
                Xp = X[va].copy()
                perm = rng.permutation(len(va))
                Xp[:, idx] = Xp[perm][:, idx]          # keep within-group structure
                drops.append(base - roc_auc_score(y[va], mod.predict_proba(Xp)[:, 1]))
            rows.append({"model": m, "group": gname, "n_cols": len(gcols),
                         "auc_drop_mean": np.mean(drops), "auc_drop_std": np.std(drops)})
    return pd.DataFrame(rows)


def time_signal(df):
    t = df["publish_date"].map(pd.Timestamp.toordinal)
    g = feature_groups(df)
    group_of = {c: k for k, v in g.items() for c in v}
    rows = []
    for c in feature_set(df, "all"):
        rho = spearmanr(df[c], t).statistic
        rows.append({"feature": c, "group": group_of[c], "spearman_with_date": rho})
    return pd.DataFrame(rows).sort_values("spearman_with_date", key=np.abs, ascending=False)


def main():
    df = load()
    y = make_label(df.shares, THRESHOLD)
    sets = feature_sets(df)

    abl = ablation(df, y, sets)
    abl.to_csv(TAB / "e3_ablation.csv", index=False)

    perm = grouped_permutation(df, y)
    perm.to_csv(TAB / "e3_grouped_permutation.csv", index=False)
    print(perm.round(4).to_string())

    ts = time_signal(df)
    ts.to_csv(TAB / "e3_time_signal.csv", index=False)
    print(ts.head(12).round(3).to_string())

    # ---- figure: ablation (HistGB, rolling-origin mean ± sd)
    order = ["all", "no_self_ref", "no_kw", "no_kw_ref", "only_kw_ref"]
    labels = ["all features", "no self_reference_*", "no kw_*", "no kw_* and\nno self_reference_*",
              "only kw_* +\nself_reference_*"]
    s = abl[abl.model == "hgb"].set_index("feature_set").loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = ["0.45", "0.65", "0.65", "C0", "0.65"]
    ax.barh(range(len(order))[::-1], s["rolling_auc_mean"], xerr=s["rolling_auc_std"],
            color=colors, capsize=2)
    for i, (v, sd) in enumerate(zip(s["rolling_auc_mean"], s["rolling_auc_std"])):
        ax.text(v + sd + 0.004, len(order) - 1 - i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(order))[::-1], labels, fontsize=8)
    ax.set_xlim(0.5, 0.78); ax.set_xlabel("ROC-AUC, gradient boosting (rolling-origin mean ± sd; 0.5 = chance)")
    fig.tight_layout(); fig.savefig(FIG / "e3_ablation.png", dpi=200)

    # ---- figure: grouped permutation importance
    p = perm[perm.model == "hgb"].sort_values("auc_drop_mean")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(p["group"], p["auc_drop_mean"], xerr=p["auc_drop_std"], capsize=2,
            color=["C0" if g in ("keyword", "self_reference") else "0.6" for g in p["group"]])
    ax.set_xlabel("drop in validation ROC-AUC when group is permuted")
    fig.tight_layout(); fig.savefig(FIG / "e3_grouped_permutation.png", dpi=200)
    print("done")


if __name__ == "__main__":
    main()
