"""Report Figures 2-4, drawn from the result tables (run the experiment scripts first).

Fig. 2 (RQ1+RQ2): (a) monthly popular share; (b) kw_max_max vs the running maximum of earlier
shares; (c) Oct-Dec 2014 AUC vs training cutoff (E1b); (d) paired ablation drops.
Fig. 3 (RQ3): threshold sweep; learning curves; regularisation at 1,000 vs 28k articles.
Fig. 4 (RQ4): popularity per KMeans cluster (k = 9) labelled by dominant channel.

Run from the repo root:  python scripts/fig_report_results.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from newspop.data import ROOT, load

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})


def fig2():
    base = pd.read_csv(TAB / "e0_monthly_pos_rate.csv", parse_dates=["publish_date"])
    mon = pd.read_csv(TAB / "e1_monthly_eval.csv")
    mon = mon[mon.n >= 1000]
    abl = pd.read_csv(TAB / "e3_ablation.csv")
    df = load()
    daily = df.groupby("publish_date")["shares"].max().sort_index()
    prior = daily.cummax().shift(1)
    kmm = df.groupby("publish_date")["kw_max_max"].median()

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.35, 1])
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[0, 1]); d = fig.add_subplot(gs[1, 1])

    a.plot(base.publish_date, base["mean"], marker="o", ms=2.5, color="0.35")
    cut = pd.Timestamp("2014-06-25")
    a.axvline(cut, color="C3", ls="--", lw=1)
    a.text(cut, 0.66, " training ends", color="C3", fontsize=7)
    a.set_ylabel("share popular"); a.set_title("(a) base rate drifts by month")
    import matplotlib.dates as mdates
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # (b) staleness vs difficulty: Oct-Dec 2014 AUC as the training cutoff moves forward
    cut_auc = pd.read_csv(TAB / "e1b_cutoff_auc.csv")
    dec = pd.read_csv(TAB / "e1b_decomposition.csv").set_index("model")
    labels = {"hgb": "HistGB", "logreg": "logistic regression"}
    for k, (m, col) in enumerate([("hgb", "C0"), ("logreg", "0.45")]):
        s_ = cut_auc[cut_auc.model == m].reset_index(drop=True)
        x = np.arange(len(s_)) + (k - 0.5) * 0.08
        b.errorbar(x, s_.auc, yerr=[s_.auc - s_.lo, s_.hi - s_.auc], marker="o", ms=3, capsize=1.5,
                   color=col, label=labels[m])
        early = dec.loc[m, "total_decay"] + s_.auc.iloc[0]
        b.plot([x[0]], [early], marker="D", ms=4, mfc="white", color=col)
    b.annotate("first model,\ntested on Jul-Sep", (0, dec.loc["hgb", "total_decay"] + cut_auc.auc.iloc[0]),
               xytext=(0.6, 0.748), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    b.set_xticks(range(4), ["26 Jun", "1 Aug", "1 Sep", "1 Oct"]); b.set_xlim(-0.4, 3.4)
    b.set_xlabel("training cutoff (2014)"); b.set_ylim(0.69, 0.764)
    b.set_ylabel("ROC-AUC on Oct-Dec 2014"); b.legend(fontsize=6.5, loc="upper right")
    b.set_title("(c) a fresher model barely helps")
    c.xaxis.set_major_locator(mdates.YearLocator())
    c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    c.step(prior.index, prior.values, where="post", color="0.65", lw=2, label="max shares of any earlier article")
    c.plot(kmm.index, kmm.values, ".", ms=1.5, color="C0", label="kw_max_max (daily median)")
    c.set_yscale("log"); c.set_ylabel("shares"); c.legend(fontsize=6.5, loc="lower right")
    c.set_title("(b) keyword statistics: earlier articles only")


    s = abl[(abl.model == "hgb") & (abl.feature_set != "all")].set_index("feature_set")
    order = ["drop_sentiment", "drop_lda", "drop_content", "drop_channel", "no_self_ref", "no_kw",
             "drop_weekday", "no_kw_ref"]
    names = {"drop_sentiment": "sentiment", "drop_lda": "LDA topics", "drop_content": "content",
             "drop_channel": "channel", "no_self_ref": "self-reference", "no_kw": "keyword",
             "drop_weekday": "weekday", "no_kw_ref": "keyword + self-ref."}
    s = s.loc[order]
    d.barh(range(len(s)), s.drop_vs_all, xerr=[s.drop_vs_all - s.drop_lo, s.drop_hi - s.drop_vs_all],
           color=["C0" if k in ("no_kw", "no_self_ref", "no_kw_ref") else "0.6" for k in s.index], capsize=1.5)
    d.axvline(0, color="0.3", lw=0.7)
    d.set_yticks(range(len(s)), [names[k] for k in s.index])
    d.set_xlabel("AUC drop when group removed (95% CI)")
    d.set_title("(d) value of each feature group (HistGB)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_time_features.png", dpi=300)


def fig3():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
    e4 = pd.read_csv(TAB / "e4_threshold.csv")
    e5 = pd.read_csv(TAB / "e5_learning_curve.csv")
    e6 = pd.read_csv(TAB / "e6_logreg_C.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7))
    s = e4[(e4.features == "all") & (e4.model == "hgb")]
    q = s["quantile"] * 100
    a.plot(q, s.roc_auc, marker="o", ms=3, color="C0", label="ROC-AUC")
    a.fill_between(q, s.auc_lo, s.auc_hi, color="C0", alpha=0.15)
    a.plot(q, s.pr_auc, marker="s", ms=3, color="0.3", label="PR-AUC")
    a.plot(q, s.pos_rate, ls=":", color="0.55", label="positive rate")
    a.set_xticks([50, 75, 90, 95]); a.set_xlabel("threshold percentile"); a.set_ylim(0, 0.8)
    a.legend(fontsize=7.5, loc="lower left"); a.set_title("(a) threshold (HistGB)")
    cols = {"logreg": "0.55", "knn": "C2", "rf": "C1", "hgb": "C0"}
    names = {"logreg": "LogReg", "knn": "kNN", "rf": "RF", "hgb": "HistGB"}
    for m in ["logreg", "knn", "rf", "hgb"]:
        d = e5[(e5.features == "all") & (e5.model == m)].groupby("n_train").test_auc.agg(["mean", "std"])
        b.errorbar(d.index, d["mean"], yerr=d["std"].fillna(0), marker="o", ms=2.5, capsize=1.5,
                   color=cols[m], label=names[m])
    b.set_xscale("log"); b.set_xlabel("training articles"); b.set_ylabel("test ROC-AUC")
    b.legend(fontsize=7.5, loc="lower right"); b.set_title("(b) learning curves")
    for size, ls_, lab in [("1000", "-", "1,000 articles"), ("full", "--", "27,750 articles")]:
        d = e6[(e6.train_size == size) & (e6.penalty == "l2")]
        c.plot(d.C, d.val_auc, ls=ls_, marker="o", ms=2.5, color="C0", label=lab)
    c.set_xscale("log"); c.set_xlabel("C (smaller = stronger L2 penalty)"); c.set_ylabel("validation ROC-AUC")
    c.legend(fontsize=7.5, loc="lower left"); c.set_title("(c) regularisation (LogReg)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_threshold_size_reg.png", dpi=300)


def fig4():
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
    p = pd.read_csv(TAB / "e7_profile_k9.csv").sort_values("popular_rate")
    share_cols = [c for c in p.columns if c.startswith("share_")]

    def label(r):
        top = r[share_cols].astype(float).idxmax()
        chan = top.replace("share_", "").replace("bus", "business").replace("socmed", "social media")
        none_share = 1 - r[share_cols].astype(float).sum()
        if none_share >= 0.5:
            chan = "no channel"
        elif r[top] < 0.5:
            chan = "mixed"
        tone = " (negative tone)" if r["mean_global_sentiment_polarity"] < 0 else ""
        return f"C{int(r['cluster'])}: {chan}, {r['dominant_topic'].replace('_', ' ')}{tone}"

    labels = [label(r) for _, r in p.iterrows()]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.barh(range(len(p)), p.popular_rate, color=["C3" if "negative" in l else "0.6" for l in labels])
    ax.axvline(0.533, color="0.3", ls=":", lw=0.8, zorder=0)
    for i, (v, n) in enumerate(zip(p.popular_rate, p.n)):
        ax.text(v + 0.005, i, f"{v:.0%} (n={n:,})", va="center", fontsize=7)
    ax.set_yticks(range(len(p)), labels, fontsize=7)
    ax.set_xlim(0.3, 1.0); ax.set_xticks([0.4, 0.6, 0.8]); ax.set_xlabel("share popular (>= 1,400)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_clusters.png", dpi=300)


if __name__ == "__main__":
    fig2(); fig3(); fig4()
