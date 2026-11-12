"""E3b (RQ2, revised) - Are the keyword statistics point-in-time, and do they carry a clock?

1. Point-in-time check: compare kw_max_max with the largest share count of any article
   published *before* the article (running maximum over the archive).
2. Clock check: Spearman correlation with publication date; spread of each column inside
   the chronological test block; kw_min_min = -1 share per year.
3. Remove the clock columns and redo (a) adversarial validation (old vs new articles) and
   (b) predictive ROC-AUC on the test block (bootstrap CI) and on rolling-origin folds
   (paired per-fold differences against the full feature set).

Run from the repo root:  python scripts/e3b_point_in_time.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from newspop.data import CLOCK_COLUMNS, CLOCK_COLUMNS_EXTRA, ROOT, feature_set, load, make_label
from newspop.evaluate import bootstrap_ci, paired_bootstrap, paired_fold_diff
from newspop.models import fit, get_model
from newspop.splits import rolling_origin, temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
SETS = ["all", "no_clock", "no_clock_strict", "no_kw", "no_kw_ref"]


def point_in_time(df):
    # largest share count among articles published on an earlier day
    daily_max = df.groupby("publish_date")["shares"].max().sort_index()
    prior = daily_max.cummax().shift(1)
    prior_max = df["publish_date"].map(prior)
    has_prior = prior_max.notna()
    kmm = df.loc[has_prior, "kw_max_max"]
    pm = prior_max[has_prior]
    stats = {
        "rows_checked": int(has_prior.sum()),
        "kw_max_max_le_prior_max": float((kmm <= pm).mean()),
        "kw_max_max_eq_prior_max": float((kmm == pm).mean()),
        "kw_max_max_gt_prior_max_rows": int((kmm > pm).sum()),
    }
    for y, g in df.groupby(df.publish_date.dt.year):
        stats[f"kw_min_min_missing_{y}"] = float(g["kw_min_min_missing"].mean())
    return stats, prior


def clock_table(df):
    n = len(df)
    _, _, te = temporal_split(n, 0.7, 0.1)
    t = df["publish_date"].map(pd.Timestamp.toordinal)
    rows = []
    for c in [c for c in df.columns if c.startswith("kw_")]:
        rows.append({"feature": c, "spearman_with_date": spearmanr(df[c], t).statistic,
                     "n_unique_in_test_block": int(df[c].iloc[te].nunique()),
                     "share_of_mode_in_test_block": float(df[c].iloc[te].value_counts(normalize=True).iloc[0]),
                     "clock": c in CLOCK_COLUMNS, "clock_extra": c in CLOCK_COLUMNS_EXTRA})
    return pd.DataFrame(rows).sort_values("spearman_with_date", key=np.abs, ascending=False)


def adversarial(df):
    n = len(df)
    z = (np.arange(n) >= int(n * 0.7)).astype(int)
    out = {}
    for fs in SETS:
        X = df[feature_set(df, fs)].to_numpy()
        p = cross_val_predict(get_model("hgb", 0), X, z, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
        out[fs] = roc_auc_score(z, p)
    return out


def predictive(df, y):
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    trv = np.concatenate([tr, va])
    folds = list(rolling_origin(n, 5, 0.4))
    test_scores, fold_auc, rows = {}, {}, []
    for m in ["logreg", "hgb"]:
        for fs in SETS:
            X = df[feature_set(df, fs)].to_numpy()
            p = fit(get_model(m, 0), X[trv], y[trv]).predict_proba(X[te])[:, 1]
            test_scores[(m, fs)] = p
            fold_auc[(m, fs)] = [roc_auc_score(y[b], fit(get_model(m, 0), X[a], y[a]).predict_proba(X[b])[:, 1])
                                 for a, b in folds]
            auc, lo, hi = bootstrap_ci(y[te], p)
            row = {"model": m, "feature_set": fs, "n_features": X.shape[1],
                   "test_auc": auc, "test_lo": lo, "test_hi": hi,
                   "rolling_mean": np.mean(fold_auc[(m, fs)])}
            if fs != "all":
                pb = paired_bootstrap(y[te], test_scores[(m, "all")], p)
                pf = paired_fold_diff(fold_auc[(m, "all")], fold_auc[(m, fs)])
                row.update({"test_drop_vs_all": pb["diff"], "test_drop_lo": pb["lo"], "test_drop_hi": pb["hi"],
                            "fold_drop_vs_all": pf["diff"], "fold_drop_lo": pf["lo"], "fold_drop_hi": pf["hi"]})
            rows.append(row)
            print(m, fs, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
    return pd.DataFrame(rows)


def main():
    df = load()
    y = make_label(df.shares, 1400)

    stats, prior = point_in_time(df)
    pd.Series(stats).to_csv(TAB / "e3b_point_in_time.csv")
    print(stats)

    ct = clock_table(df)
    ct.to_csv(TAB / "e3b_clock_columns.csv", index=False)
    print(ct.round(3).to_string())

    adv = adversarial(df)
    pd.Series(adv, name="adversarial_auc").to_csv(TAB / "e3b_adversarial.csv")
    print("adversarial:", {k: round(v, 3) for k, v in adv.items()})

    pred = predictive(df, y)
    pred.to_csv(TAB / "e3b_predictive.csv", index=False)

    # ---- staircase figure
    daily = df.groupby("publish_date").agg(kw_max_max=("kw_max_max", "median"))
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.step(prior.index, prior.values, where="post", color="0.6", lw=2,
            label="largest share count of any earlier article")
    ax.plot(daily.index, daily["kw_max_max"], ".", ms=2.5, color="C0",
            label="kw_max_max (daily median)")
    ax.set_yscale("log"); ax.set_ylabel("shares")
    ax.legend(fontsize=7, loc="lower right")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "e3b_staircase.png", dpi=200)
    print("done")


if __name__ == "__main__":
    main()
