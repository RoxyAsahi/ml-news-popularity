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
    for model, col, mk, name in [("hgb_all", ACC, "o-", "HistGB"), ("logreg_all", GREY, "s-", "L1 logistic")]:
        r = R[R.model == model]
        c.plot(r.coverage, r.accuracy, mk, color=col, ms=3.5 if mk == "o-" else 3, label=name)
    c.legend(frameon=False, fontsize=7)
    c.set_xscale("log")
    c.set_xticks([0.05, 0.1, 0.2, 0.4, 1.0], ["5%", "10%", "20%", "40%", "100%"])
    c.set_xlabel("share of articles predicted")
    c.set_ylabel("accuracy")
    c.set_title("(c) Selective prediction")
    fig.tight_layout()
    fig.savefig(FIG / "v2_fig_c_context_uncertainty.png", dpi=300)


def fig_d():
    T = pd.read_csv(TAB / "s1b_twin_precision.csv")
    T = T[T.subset.str.startswith("band")]
    bands = sorted(T.subset.unique())
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(bands))
    for filt, col, mk, off, name in [(False, GREY, "s", -0.1, "all pairs"), (True, ACC, "o", 0.1, "series removed")]:
        t = T[T.series_filtered == filt].set_index("subset").loc[bands]
        ax.errorbar(x + off, t.precision, yerr=[t.precision - t.lo, t.hi - t.precision], fmt=mk + "-", ms=3.5, color=col, label=name, lw=1)
    ax.axvline(1.5, color="0.7", ls="--", lw=0.8)
    ax.text(1.6, 0.12, "threshold used", fontsize=7, color="0.4")
    ax.set_xticks(x, [b.replace("band ", "").replace("[", "").replace(")", "").replace(",", "-") for b in bands], fontsize=7)
    ax.set_xlabel("slug cosine similarity band")
    ax.set_ylabel("share same story")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "v2_fig_d_twin_precision.png", dpi=300)


def fig_e():
    """Benchmark sanity checks: learning curve (E5) and label threshold (E4), chronological test block."""
    L = pd.read_csv(TAB / "e5_learning_curve_summary.csv", header=[0, 1], index_col=[0, 1, 2])
    T = pd.read_csv(TAB / "e4_threshold.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 2.3))
    styles = {"hgb": (ACC, "o-", "HistGB"), "rf": ("#d9822b", "^-", "RF"), "logreg": (GREY, "s-", "L1 logistic"),
              "knn": ("0.3", "x:", "kNN")}
    for m, (col, mk, name) in styles.items():
        r = L.loc[("all", m)]
        a.plot(r.index, r[("test_auc", "mean")], mk, color=col, ms=2.5, lw=1, label=name)
    a.set_xscale("log")
    a.set_xlabel("training articles")
    a.set_ylabel("test ROC-AUC")
    a.set_title("(a) Learning curve")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    for m, (col, mk, name) in [(k, styles[k]) for k in ["hgb", "logreg"]]:
        r = T[(T.features == "all") & (T.model == m)]
        xq = np.arange(len(r))
        b.plot(xq, r.roc_auc, mk, color=col, ms=2.5, lw=1, label="_")
        b.plot(xq, r.pr_auc, mk.replace("-", "--"), color=col, ms=2.5, lw=1, label="_")
    b.set_xticks(range(4), ["50", "75", "90", "95"])
    b.set_xlabel("threshold percentile")
    b.set_title("(b) Viral threshold")
    b.plot([], [], "-", color="0.2", lw=1, label="ROC-AUC")
    b.plot([], [], "--", color="0.2", lw=1, label="PR-AUC")
    b.set_ylim(0, 0.8)
    b.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.66))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "v2_fig_e_benchmark.png", dpi=300)


def fig_f():
    """Upworthy design ladder (confirmatory split): (a) features with detectable bias and coverage of the
    randomized truth; (b) per-feature estimates against the randomized truth."""
    R = pd.read_csv(TAB / "u1_ladder_confirmatory.csv")
    C = pd.read_csv(TAB / "u1_coefs_confirmatory.csv", index_col=0)
    short = {"L0 naive cross-section": "cross-section", "L1 + week FE": "+ week FE",
             "L1e + week FE + text embedding": "+ week FE\n+ text embedding", "L2 story twins": "story twins",
             "L3 same-article retests": "same-article\nretests"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 2.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    yy = np.arange(len(R))
    cols = [GREY, GREY, GREY, ACC, ACC]
    a.barh(yy, R.n_features_biased, color=cols, height=0.6)
    for k, (nb, cv) in enumerate(zip(R.n_features_biased, R.coverage_of_truth)):
        a.text(nb + 0.3, k, f"{cv:.0%}", va="center", fontsize=6.5)
    a.set_yticks(yy, [short[d] for d in R.design], fontsize=6.5)
    a.invert_yaxis()
    a.set_xlim(0, 14)
    a.set_xlabel("features (of 15) with\ndetectable bias")
    a.set_title("(a) Biased features", fontsize=7.5)
    t = C["T truth (randomized)"]
    for d, col, mk, lab in [("L1 + week FE", GREY, "s", "+ week FE"),
                            ("L3 same-article retests", ACC, "o", "retests")]:
        b.scatter(t, C[d], s=10, color=col, marker=mk, label=lab, zorder=3)
    lim = [-0.07, 0.09]
    b.plot(lim, lim, color="0.7", lw=0.8, ls="--")
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("randomized effect", fontsize=7)
    b.set_ylabel("observational estimate", fontsize=7)
    b.tick_params(labelsize=6)
    b.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2)
    b.set_title("(b) vs. truth", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "v2_fig_f_upworthy.png", dpi=300)


def fig_g():
    """Upworthy, confirmatory split (u6/u7, u5): (a) decision accuracy against the number of randomized tests,
    with archives as reference lines; (b) exchange rate against archive size, idealized vs published archive;
    (c) two validation metrics against decision accuracy for the seven GloVe models."""
    R = pd.read_csv(TAB / "u6_runs_confirmatory.csv")
    X = pd.read_csv(TAB / "u7_exchange_rate.csv").query("split == 'confirmatory'")
    F = pd.read_csv(TAB / "u7_fusion.csv").query("split == 'confirmatory'")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    for m, arch, col, lab in [("E", "-", "0.25", "tests only"), ("PRIOR", "random", ACC, "idealized archive as prior"),
                              ("PRIOR", "published", "#d08c2b", "published archive as prior")]:
        r = F[(F.method == m) & (F.archive == arch) & (F.k > 0)].sort_values("k")
        a.plot(r.k, r.acc, marker="o", ms=2.3, color=col, lw=1.1, label=lab)
    for arch, ls_, col in [("random", "--", ACC), ("published", "--", "#d08c2b")]:
        v = X[(X.method == "O") & (X.archive == arch) & (X["size"] == -1)].acc.iloc[0]
        a.axhline(v, color=col, ls=ls_, lw=0.9)
    v = X[(X.method == "O-raw") & (X.archive == "random") & (X["size"] == -1)].acc.iloc[0]
    a.axhline(v, color=GREY, ls=":", lw=0.9)
    a.text(27, v - 0.006, "idealized archive, raw CTR", fontsize=5.5, color="0.35")
    a.plot([], [], color="0.5", ls="--", lw=0.9, label="archive alone (dashed)")
    a.set_xscale("log")
    a.set_xlabel("randomized tests used for training")
    a.set_ylabel("accuracy, within-test pairs")
    a.set_ylim(0.56, 0.70)
    a.legend(frameon=False, fontsize=5.5, loc="lower right")
    a.set_title("(a) archives vs. randomized tests", fontsize=7.5)
    n_all = 18194
    for arch, col, lab in [("random", ACC, "idealized"), ("published", "#d08c2b", "published")]:
        r = X[(X.method == "O") & (X.archive == arch)].copy()
        r["n"] = r["size"].where(r["size"] > 0, n_all)
        r = r.sort_values("n")
        hi = r.kstar_hi.replace(np.inf, 12000)
        b.errorbar(r.n, r.kstar, yerr=[r.kstar - r.kstar_lo, hi - r.kstar], fmt="o-", ms=3, color=col, lw=1, label=lab,
                   capsize=0)
    law = pd.read_csv(TAB / "u9_exchange_law.csv").set_index("archive")
    nn = np.logspace(np.log10(800), np.log10(40000), 60)
    for arch, col in [("random", ACC), ("published", "#d08c2b")]:
        K, n0 = law.loc[arch, "K"], law.loc[arch, "n0"]
        b.plot(nn, K * nn / (nn + n0), color=col, ls=":", lw=0.8)
        b.axhline(K, color=col, ls=(0, (1, 3)), lw=0.6)
        Xo = pd.read_csv(TAB / "u7_exchange_rate.csv").query("split != 'confirmatory' and method == 'O' and archive == @arch and size == -1")
        b.scatter([law.loc[arch, "holdout_n"]] * len(Xo), Xo.kstar, facecolors="none", edgecolors=col, s=14, lw=0.8,
                  zorder=4)
    b.scatter([], [], facecolors="none", edgecolors="0.4", s=14, label="smaller splits")
    b.plot([], [], color="0.4", ls=":", lw=0.8, label="fit $K n/(n+n_0)$")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("archive size (headlines)")
    b.set_ylabel("worth in randomized tests")
    b.legend(frameon=False, fontsize=5.5, loc="lower right")
    b.set_title("(b) exchange rate", fontsize=7.5)
    M = pd.read_csv(TAB / "u5_observational_metrics.csv").query("split == 'confirmatory' and within_acc_sig == within_acc_sig")
    for kind, col, mk, lab in [("raw", GREY, "s", "CTR across tests"), ("week", ACC, "o", "CTR demeaned in week")]:
        g = M[M.metric == kind]
        c.scatter(g.spearman, g.within_acc_sig, color=col, marker=mk, s=12, label=lab, zorder=3)
        o = g[g.train == "ORACLE"]
        c.scatter(o.spearman, o.within_acc_sig, facecolors="none", edgecolors="k", s=40, lw=0.8, zorder=4)
    c.annotate("randomized\ntraining", (M[(M.metric == "week") & (M.train == "ORACLE")].spearman.iloc[0], 0.6841),
               xytext=(-30, -18), textcoords="offset points", fontsize=5.5, color="0.3")
    c.set_xlabel("held-out Spearman with CTR")
    c.set_ylabel("decision accuracy")
    c.legend(frameon=False, fontsize=5.5, loc="lower right")
    c.set_title("(c) validation vs. decision", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "v2_fig_g_archive.png", dpi=300)


import sys
if len(sys.argv) > 1:
    globals()[sys.argv[1]]()
else:
    fig_a(); fig_b(); fig_c(); fig_d(); fig_e(); fig_g()
print("figures written")
