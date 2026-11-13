"""Figure for v19 RQ1-2: where the day's top pick lands, where the skill lives, which signals carry it.
Reads data/interim/v1_scores.csv, explore/jev/results/api_lineup.csv, results/tables/s2_anatomy.csv and
results/tables/v3_signal_groups.csv. Output: results/figures/v19_fig_called.png"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402
T = ROOT / "results" / "tables"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(width_ratios=[1, 1.05, 1.15]))

# (a) decile of the day (by shares) in which the top-scored article lands
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy())
dn = (pd.to_datetime(df.publish_date).dt.normalize() - pd.Timestamp("2013-01-01")).dt.days.to_numpy()
S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv"); S = S[S.has_title]
L = pd.read_csv(ROOT / "explore" / "jev" / "results" / "api_lineup.csv")
def deciles(rows, score):
    out = []
    for d, g in pd.DataFrame({"r": rows, "s": score, "d": dn[rows]}).groupby("d"):
        if len(g) < 10: continue
        yy = y[g.r.to_numpy()]; top = yy[np.argmax(g.s.to_numpy())]
        out.append(int(np.sum(top >= np.quantile(yy, np.arange(1, 10) / 10))))  # 9 = top tenth (>= 90th pct, as in Table 1)
    return np.bincount(out, minlength=10) / len(out)
x = np.arange(1, 11)
for lab, rows, s, col, ls in [("HistGB (past-only)", S.row.to_numpy(), S.hgb.to_numpy(), BLUE, "-"),
                              ("kw_avg_avg alone", S.row.to_numpy(), S.kw_avg_avg.to_numpy(), AQUA, "-"),
                              ("JEV judge, headlines", L.row.to_numpy(), L.p_top.to_numpy(), ORANGE, "-")]:
    a.plot(x, 100 * deciles(rows, s), ls, color=col, lw=1.8, marker="o", ms=3, label=lab)
a.axhline(10, color=MUTED, lw=0.8, ls="--"); a.text(1.1, 11.2, "random pick", fontsize=6.5, color=MUTED)
a.set_xticks(x, ["low", "", "", "", "", "", "", "", "", "top"]); a.set_xlabel("tenth of the day by shares")
a.set_ylabel("% of days the top pick lands there"); a.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=1.5)
a.set_ylim(0, 50); a.spines[["top", "right"]].set_visible(False); a.set_title("(a) The day's top pick", loc="left", fontsize=8)

# (b) within-context AUC ladder
A = pd.read_csv(T / "s2_anatomy.csv")
order = ["all pairs", "same day", "same channel", "same channel x day", "same-story twins, same channel"]
labels = ["all pairs", "same day", "same channel", "same channel & day", "same story & channel"]
yy = np.arange(len(order))[::-1]
for mname, lab, col, off in [("hgb_all", "all features", BLUE, 0.13), ("hgb_noshare", "without history", MUTED, -0.13)]:
    r = A[A.model == mname].set_index("stratum").loc[order]
    b.errorbar(r.auc, yy + off, xerr=[r.auc - r.lo, r.hi - r.auc], fmt="o", ms=3.5, color=col, lw=1.2, capsize=0, label=lab)
b.axvline(0.5, color=MUTED, lw=0.8, ls="--"); b.set_yticks(yy, labels); b.tick_params(axis="y", length=0)
b.set_xlim(0.48, 0.78); b.set_xlabel("pairwise AUC within context"); b.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=1.2)
b.spines[["top", "right"]].set_visible(False); b.set_title("(b) Where the skill lives", loc="left", fontsize=8)

# (c) signal groups: accuracy on same-day same-section pairs
G = pd.read_csv(T / "v3_signal_groups.csv"); G = G[G.metric == "pair_sec"].set_index("features")
groups = ["history", "body", "title", "placement"]; names = ["history", "body", "title features", "channel, weekday"]
yy = np.arange(len(groups))[::-1]
for kind, col, off, lab in [("only", AQUA, 0.13, "this group only"), ("without", BLUE, -0.13, "all but this group")]:
    r = G.loc[[f"{kind}_{g}" for g in groups]]
    c.errorbar(100 * r.est, yy + off, xerr=[100 * (r.est - r.lo), 100 * (r.hi - r.est)], fmt="o", ms=3.5, color=col, lw=1.2, label=lab)
allv = 100 * G.loc["all", "est"]; c.axvline(allv, color=BLUE, lw=0.8, ls=":"); c.text(allv + 0.2, yy[-1] - 0.45, "all features", fontsize=6.5, color=BLUE)
c.axvline(50, color=MUTED, lw=0.8, ls="--"); c.set_yticks(yy, names); c.tick_params(axis="y", length=0)
c.set_xlabel("% correct, same day & section"); c.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=1.2)
c.set_xlim(49, 63); c.spines[["top", "right"]].set_visible(False); c.set_title("(c) Which signals", loc="left", fontsize=8)
fig.tight_layout(w_pad=0.6); fig.savefig(ROOT / "results" / "figures" / "v19_fig_called.png", dpi=300); print("saved")
