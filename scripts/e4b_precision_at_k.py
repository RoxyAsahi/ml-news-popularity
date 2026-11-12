"""E4b (RQ3) - The editor's view at the 95th-percentile threshold.

An editor can promote only a few articles a week. Train HistGB on the training block with
"viral" = shares >= the training block's 95th percentile, then, in every week of the test
block, flag the top 5% of articles by predicted score and measure the precision of that
shortlist (share of flagged articles that were really viral) against the weekly base rate.

Run from the repo root:  python scripts/e4b_precision_at_k.py   (~30 s)
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from newspop.data import ROOT, feature_set, load, make_label
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
Q, TOP = 0.95, 0.05


def main():
    df = load()
    n = len(df)
    tr, va, te = temporal_split(n, 0.7, 0.1)
    thr = df.shares.iloc[tr].quantile(Q)
    y = make_label(df.shares, thr)
    X = df[feature_set(df, "all")].to_numpy()
    p = fit(get_model("hgb", 0), X[tr], y[tr]).predict_proba(X[te])[:, 1]
    t = pd.DataFrame({"week": df.publish_date.iloc[te].dt.to_period("W").dt.start_time.to_numpy(),
                      "y": y[te], "p": p})
    rows = []
    for w, g in t.groupby("week"):
        k = max(1, int(np.ceil(TOP * len(g))))
        top = g.nlargest(k, "p")
        rows.append({"week": w, "n": len(g), "k": k, "hits": int(top.y.sum()),
                     "precision": top.y.mean(), "base_rate": g.y.mean()})
    wk = pd.DataFrame(rows)
    wk = wk[wk.n >= 100]                       # drop partial first/last weeks
    hits, flagged = wk.hits.sum(), wk.k.sum()
    rng = np.random.default_rng(0)
    boots = [wk.iloc[i].hits.sum() / wk.iloc[i].k.sum() for i in
             (rng.integers(0, len(wk), len(wk)) for _ in range(1000))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    summary = pd.DataFrame([{"threshold": thr, "weeks": len(wk), "flagged": flagged, "hits": hits,
                             "precision": hits / flagged, "lo": lo, "hi": hi,
                             "base_rate": wk.eval("base_rate * n").sum() / wk.n.sum(),
                             "weeks_zero_hits": int((wk.hits == 0).sum())}])
    summary["lift"] = summary.precision / summary.base_rate
    wk.to_csv(TAB / "e4b_weekly_precision.csv", index=False)
    summary.to_csv(TAB / "e4b_precision_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(wk.week, wk.precision, width=5, color="C0", label="precision of weekly top 5%")
    ax.plot(wk.week, wk.base_rate, color="0.3", ls=":", marker=".", ms=3, label="base rate")
    ax.set_ylabel("share viral"); ax.set_ylim(0, 0.6)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("week (2014)")
    fig.tight_layout(); fig.savefig(FIG / "e4b_precision_at_top5.png", dpi=300)


if __name__ == "__main__":
    main()
