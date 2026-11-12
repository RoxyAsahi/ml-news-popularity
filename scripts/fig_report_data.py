"""Report Figure 1 and Table 1 inputs (Data section).

(a) distribution of shares (log scale) with the 1,400 threshold;
(b) median and inter-quartile range of shares by article age (days before collection).

Run from the repo root:  python scripts/fig_report_data.py
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a.hist(np.log10(s), bins=70, color="0.55")
    a.axvline(np.log10(1400), color="C0", ls="--", lw=1.2)
    a.text(np.log10(1400) + 0.05, a.get_ylim()[1] * 0.92, "1,400 (median)", color="C0", fontsize=7)
    a.set_xlabel("shares (log$_{10}$)"); a.set_ylabel("articles"); a.set_title("(a) shares", fontsize=9)

    bins = [7, 30, 60, 120, 240, 360, 480, 600, 731]
    g = df.groupby(pd.cut(df["timedelta"], bins), observed=True)["shares"]
    q = g.quantile([0.25, 0.5, 0.75]).unstack()
    mids = [(i.left + i.right) / 2 for i in q.index]
    b.fill_between(mids, q[0.25], q[0.75], color="0.85", label="inter-quartile range")
    b.plot(mids, q[0.5], marker="o", ms=3, color="C0", label="median")
    rho = spearmanr(df["timedelta"], s).statistic
    b.set_xlabel("article age at collection (days)"); b.set_ylabel("shares")
    b.set_title(f"(b) shares vs age (Spearman $\\rho$ = {rho:.2f})", fontsize=9)
    b.set_ylim(0, 4000); b.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_data.png", dpi=300)

    groups = feature_groups(df)
    pd.DataFrame([{"group": k, "n_columns": len(v), "columns": ", ".join(v)} for k, v in groups.items()]) \
        .to_csv(TAB / "table1_feature_groups.csv", index=False)


if __name__ == "__main__":
    main()
