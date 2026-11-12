"""E0 - exploratory analysis. Writes figures to results/figures and a summary table.

Run from the repo root:  python scripts/00_eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from newspop.data import ROOT, feature_groups, load

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def main():
    df = load()
    s = df["shares"]
    print(f"rows={len(df)}  dates {df.publish_date.min().date()} .. {df.publish_date.max().date()}")
    print(s.describe(percentiles=[.25, .5, .75, .9, .95, .99]).round(1))

    # 1. heavy tail
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.log10(s), bins=80, color="0.4")
    ax.axvline(np.log10(1400), color="C0", ls="--", label="median = 1400")
    ax.set_xlabel("log10(shares)"); ax.set_ylabel("articles"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "e0_shares_hist.png", dpi=200)

    # 2. do shares keep growing with article age?
    rho = spearmanr(df["timedelta"], s)
    print(f"Spearman(timedelta, shares) = {rho.statistic:.3f} (p={rho.pvalue:.1e})")
    bins = pd.cut(df["timedelta"], [0, 30, 60, 120, 240, 400, 600, 800])
    by_age = df.groupby(bins, observed=True)["shares"].agg(["count", "median", "mean"])
    by_age.to_csv(TAB / "e0_shares_by_age.csv")
    print(by_age)

    # 3. popularity rate per month (base-rate drift, RQ1)
    monthly = df.set_index("publish_date")["shares"].ge(1400).resample("MS").agg(["mean", "size"])
    monthly.to_csv(TAB / "e0_monthly_pos_rate.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(monthly.index, monthly["mean"], marker="o", color="C0")
    ax.axhline(0.5, color="0.6", ls=":")
    ax.set_ylabel("share of articles with >= 1400 shares"); ax.set_xlabel("publication month")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "e0_monthly_pos_rate.png", dpi=200)

    # 4. channel / weekday medians
    g = feature_groups(df)
    rows = []
    for c in g["channel"] + g["weekday"]:
        rows.append({"feature": c, "n": int(df[c].sum()), "median_shares": df.loc[df[c] == 1, "shares"].median()})
    pd.DataFrame(rows).to_csv(TAB / "e0_group_medians.csv", index=False)

    # data-quality counts for the report
    print("empty content:", int(df.empty_content.sum()),
          "| kw_min_min missing:", f"{df.kw_min_min_missing.mean():.1%}")


if __name__ == "__main__":
    main()
