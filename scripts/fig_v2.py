"""v2 report figures, drawn from result tables (run s2-s7 first).

Fig. A (RQ1): pairwise AUC within nested contexts (past-only HistGB and L1 logistic regression).
Fig. B (RQ1+RQ2): (a) twin correlation vs matched random neighbours; (b) content effects,
                  cross-section vs within twins.
Fig. C (RQ3+RQ4): (a) daily-volume coefficient across specifications; (b) monthly coverage of
                  80% intervals; (c) risk-coverage of selective prediction.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from newspop.data import ROOT

FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.spines.top": False, "axes.spines.right": False})
ACC, GREY = "#1f5fbf", "0.55"


def fig_a():
    R = pd.read_csv(TAB / "s2_anatomy.csv")
    order = ["all pairs", "same week", "same day", "same channel", "same channel x day",
             "same-story twins", "same-story twins, same channel"]
    labels = ["all pairs", "same week", "same day", "same channel", "same channel & day",
              "same-story twins", "same-story twins,\nsame channel"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for k, (model, col, off, name) in enumerate([("hgb_all", ACC, -0.13, "HistGB"), ("logreg_all", GREY, 0.13, "L1 logistic")]):
        r = R[R.model == model].set_index("stratum").loc[order]
        yy = np.arange(len(order)) + off
        ax.errorbar(r.auc, yy, xerr=[r.auc - r.lo, r.hi - r.auc], fmt="o", ms=3.5, color=col, lw=1.2, label=name)
    ax.axvline(0.5, color="0.7", ls="--", lw=0.8)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.48, 0.8)
    ax.set_xlabel("pairwise ROC-AUC within context\n(0.5 = chance)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "v2_fig_a_anatomy.png", dpi=300)


def fig_b():
    S = pd.read_csv(TAB / "s3_story_similarity.csv")
    E = pd.read_csv(TAB / "s4_twin_effects.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    subs = ["all twins (sim >= 0.5)", "same day", "1-2 days apart", "3-7 days apart", "same channel", "different channel"]
    s = S.set_index("subset").loc[subs]
    yy = np.arange(len(subs))
    a.errorbar(s.r_twin, yy, xerr=[s.r_twin - s.r_twin_lo, s.r_twin_hi - s.r_twin], fmt="o", ms=3.5, color=ACC, label="twin pairs")
    a.plot(s.r_matched_random, yy, "x", color=GREY, ms=5, label="matched random\nneighbour")
    a.axvline(0, color="0.7", lw=0.8)
    a.set_yticks(yy, ["all twins", "same day", "1-2 days apart", "3-7 days apart", "same channel", "different channel"])
    a.invert_yaxis()
    a.set_xlabel("correlation of residual log shares")
    a.set_title("(a) Do same-story articles share their fate?")
    a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    e = E[E.variant == "main (sim >= 0.5)"].dropna(subset=["cross_section"]).set_index("feature")
    e = e.reindex(e.twin.sort_values().index)
    yy = np.arange(len(e))
    b.errorbar(e.cross_section, yy - 0.15, xerr=[e.cross_section - e.cs_lo, e.cs_hi - e.cross_section], fmt="s", ms=3, color=GREY, label="cross-section (channel x week FE)")
    b.errorbar(e.twin, yy + 0.15, xerr=[e.twin - e.tw_lo, e.tw_hi - e.twin], fmt="o", ms=3.5, color=ACC, label="within twins")
    b.axvline(0, color="0.7", lw=0.8)
    b.set_yticks(yy, e.index)
    b.set_xlabel("effect of +1 sd on log shares")
    b.set_title("(b) Content effects with the story held fixed")
    b.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "v2_fig_b_twins.png", dpi=300)


def fig_c():
    C = pd.read_csv(TAB / "s5_competition.csv")
    V = pd.read_csv(TAB / "s7_coverage.csv", index_col=0)
    R = pd.read_csv(TAB / "s7_selective.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    keep = ["main: week + weekday + channel FE", "+ all content features", "long articles (>= 600 words)",
            "drop 5% busiest days", "year 2013", "year 2014", "within twins (story-clustered)",
            "placebo: volume one week later"]
    names = ["main (week, weekday,\nchannel FE)", "+ content features", "long articles only", "drop busiest 5% days",
             "2013 only", "2014 only", "within twins", "placebo: next week's\nvolume"]
    c_ = C.set_index("spec").loc[keep]
    yy = np.arange(len(keep))
    cols = [GREY if "placebo" in k else ACC for k in keep]
    for y_, (_, r), col in zip(yy, c_.iterrows(), cols):
        a.errorbar(r.coef, y_, xerr=[[r.coef - r.lo], [r.hi - r.coef]], fmt="o", ms=3.5, color=col)
    a.axvline(0, color="0.7", lw=0.8)
    a.set_yticks(yy, names, fontsize=7)
    a.invert_yaxis()
    a.set_xlabel("log shares per log daily volume")
    a.set_title("(a) Same-day competition")
    v = V.drop(index="all test")
    x = np.arange(len(v))
    b.plot(x, v["CQR"], "s-", color=GREY, ms=3, label="CQR")
    b.plot(x, v["CQR + ACI"], "o-", color=ACC, ms=3.5, label="CQR + ACI")
    b.axhline(0.8, color="0.7", ls="--", lw=0.8)
    b.set_xticks(x, [m[2:] for m in v.index], rotation=45)
    b.set_ylim(0.72, 0.9)
    b.set_ylabel("coverage of 80% interval")
    b.set_title("(b) Intervals under drift")
    b.legend(frameon=False, fontsize=7)
    c.plot(R.coverage, R.accuracy, "o-", color=ACC, ms=3.5)
    c.set_xscale("log")
    c.set_xticks([0.05, 0.1, 0.2, 0.4, 1.0], ["5%", "10%", "20%", "40%", "100%"])
    c.set_xlabel("share of articles predicted")
    c.set_ylabel("accuracy")
    c.set_title("(c) Selective prediction")
    fig.tight_layout()
    fig.savefig(FIG / "v2_fig_c_context_uncertainty.png", dpi=300)


fig_a(); fig_b(); fig_c()
print("figures written")
