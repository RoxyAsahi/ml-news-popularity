"""Figures for the v19 report. Reads results/tables/v1_*, v2_judge_summary.csv and explore/jev/results.
Output: results/figures/v19_fig_judge.png"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402

T = ROOT / "results" / "tables"; J = ROOT / "explore" / "jev" / "results"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"
V1 = pd.read_csv(T / "v1_past_only_called.csv"); V2 = pd.read_csv(T / "v2_judge_summary.csv")


def v1(scorer, metric="pair_sec"):
    r = V1[(V1.scorer == scorer) & (V1.metric == metric)].iloc[0]; return r.est, r.lo, r.hi


def v2(setting, scorer, metric="acc"):
    r = V2[(V2.setting == setting) & (V2.scorer == scorer) & (V2.metric == metric)]
    return None if r.empty else (r.iloc[0].est, r.iloc[0].lo, r.iloc[0].hi)


def rep(name):
    R = pd.read_csv(J / f"rep_curve_confirmatory_{name}.csv"); R = R[R.k == -1]
    return (R.within_acc_sig * R.n_sig).sum() / R.n_sig.sum(), np.nan, np.nan


MS, UP = "mashable_same_section_pairs", "upworthy_sig_pairs"
rows = [("Across Mashable articles, same day & section", None, None),
        ("HistGB, 58 features", v1("hgb"), BLUE),
        ("kw_avg_avg alone (keyword history)", v1("kw_avg_avg"), BLUE),
        ("HistGB + headline embedding", v1("hgb_bge"), BLUE),
        ("Ridge, TF-IDF of headline", v1("tfidf"), BLUE),
        ("Ridge, bge headline embedding", v1("bge"), BLUE),
        ("JEV judge, headline", v2(MS, "judge_api"), BLUE),
        ("... its most confident 10%", v2(MS, "judge_api", "acc_top10pct_conf"), BLUE),
        ("JEV-9B (open), headline", v2(MS + "_scored_months", "judge_open_9b"), BLUE),
        ("JEV judge, headline + 6 features", v2("mashable_same_section_pairs_featprompt", "judge_api_title_plus_features"), BLUE),
        ("Within one Upworthy article, randomized", None, None),
        ("JEV judge, headline", v2(UP, "judge_api"), ORANGE),
        ("... its most confident 25%", v2(UP, "judge_api", "acc_top25pct_conf"), ORANGE),
        ("JEV-9B (open), headline", v2(UP, "judge_open_9b"), ORANGE),
        ("Ridge on RoBERTa, trained on tests", rep("roberta"), ORANGE),
        ("Ridge on GloVe, trained on tests", rep("glove"), ORANGE)]
rows = [r for r in rows if r[2] is None or r[1] is not None]

plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1.0, 1]))
ys = np.arange(len(rows))[::-1]
for yv, (lab, val, col) in zip(ys, rows):
    if val is None:
        a.axhspan(yv - 0.5, yv + 0.5, color="#f1f0ec", lw=0, zorder=0); continue
    est, lo, hi = val
    if not np.isnan(lo):
        a.plot([lo, hi], [yv, yv], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    a.plot(est, yv, "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=3)
    a.text(0.985, yv, f"{100 * est:.1f}", fontsize=7, va="center", ha="right", color=INK)
a.axvline(0.5, color=MUTED, lw=0.8, ls="--", zorder=1)
a.set_yticks(ys, [r[0] if r[1] is not None else r[0] for r in rows])
for t, r in zip(a.get_yticklabels(), rows):
    if r[1] is None:
        t.set_fontweight("bold")
a.tick_params(axis="y", length=0)
a.set_xlim(0.45, 0.99); a.set_ylim(-0.6, len(rows) - 0.4)
a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); a.set_xlabel("pairwise accuracy (50 = chance)")
a.spines[["top", "right", "left"]].set_visible(False); a.set_title("(a) Calling the better of two", loc="left", fontsize=8)

# (b) accuracy as a function of how many pairs the scorer keeps (most confident first)
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy())
def curve(p, truth):
    o = np.argsort(-np.abs(p - .5) if p.max() <= 1 else -np.abs(p), kind="stable"); c = ((p > (.5 if p.max() <= 1 else 0)) == truth)[o]
    k = np.arange(1, len(c) + 1); keep = k / len(c); acc = np.cumsum(c) / k; m = keep >= 0.05
    return keep[m], acc[m]
U = pd.read_csv(J / "api_upw_pairs.csv"); pu = ((U.p0 + 1 - U.p1) / 2).to_numpy()
x, yy = curve(pu, U.zz.to_numpy() > 0); b.plot(x, yy, color=ORANGE, lw=2, label="JEV, Upworthy (within article)")
P = pd.read_csv(J / "api_mash_titles_section.csv"); pm = ((P.p0_title + 1 - P.p1_title) / 2).to_numpy(); tm = y[P.a] > y[P.b]
x, yy = curve(pm, tm); b.plot(x, yy, color=BLUE, lw=2, label="JEV, Mashable (across articles)")
S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv"); h = np.full(len(df), np.nan); h[S.row] = S.hgb
ok = ~np.isnan(h[P.a]) & ~np.isnan(h[P.b]); dh = (h[P.a] - h[P.b])[ok]
x, yy = curve(dh, tm[ok]); b.plot(x, yy, color=BLUE, lw=1.5, ls=(0, (2.5, 1.5)), label="HistGB on features, Mashable")
b.axhline(0.5, color=MUTED, lw=0.8, ls="--")
b.set_xlabel("share of pairs kept (most confident first)"); b.set_ylabel("accuracy on kept pairs")
b.set_xlim(0, 1); b.set_ylim(0.32, 0.95); b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9]); b.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=1)
b.spines[["top", "right"]].set_visible(False); b.set_title("(b) Does confidence help?", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "v19_fig_judge.png", dpi=300)
print("saved")
