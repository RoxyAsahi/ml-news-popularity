"""Figure for RQ2: where the day's top pick lands, and which signals carry the skill (repaired data).
Reads data/interim/v1_scores.csv, explore/jev/results/rep_lineup.csv, results/tables/v1_past_only_called.csv and
results/tables/v3_signal_groups.csv. Output: results/figures/fig_called.png"""
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
fig, (a, c) = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15]))

# (a) tenth of the day (by shares) in which the top-scored article lands
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy())
dn = (pd.to_datetime(df.publish_date).dt.normalize() - pd.Timestamp("2013-01-01")).dt.days.to_numpy()
S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv"); S = S[S.has_title]
L = pd.read_csv(ROOT / "explore" / "jev" / "results" / "rep_lineup.csv"); L = L[np.isin(L.row, S.row)]


def deciles(rows, score):
    out = []
    for d, g in pd.DataFrame({"r": rows, "s": score, "d": dn[rows]}).groupby("d"):
        if len(g) < 10:
            continue
        yy = y[g.r.to_numpy()]; top = yy[np.argmax(g.s.to_numpy())]
        out.append(int(np.sum(top >= np.quantile(yy, np.arange(1, 10) / 10))))  # 9 = top tenth (>= 90th pct, as in Table 1)
    return np.bincount(out, minlength=10) / len(out)


x = np.arange(1, 11)
for lab, rows, s, col, ls in [("58 features + headline", S.row.to_numpy(), S.hgb_bge.to_numpy(), INK, "-"),
                              ("58 features (HistGB)", S.row.to_numpy(), S.hgb.to_numpy(), BLUE, "-"),
                              ("headline only (ridge)", S.row.to_numpy(), S.bge.to_numpy(), ORANGE, "-"),
                              ("zero-shot judge, headlines", L.row.to_numpy(), L.p_top.to_numpy(), ORANGE, (0, (2, 1.5)))]:
    a.plot(x, 100 * deciles(rows, s), ls=ls, color=col, lw=1.6, marker="o", ms=2.8, label=lab)
a.axhline(10, color=MUTED, lw=0.8, ls="--"); a.text(1.1, 11.2, "random pick", fontsize=6.5, color=MUTED)
a.set_xticks(x, ["low", "", "", "", "", "", "", "", "", "top"]); a.set_xlabel("tenth of the day by shares")
a.set_ylabel("% of days the top pick lands there"); a.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=1.8)
a.set_ylim(0, 50); a.spines[["top", "right"]].set_visible(False); a.set_title("(a) The day's top pick", loc="left", fontsize=8)

# (b) signals: accuracy on same-day same-section pairs
G = pd.read_csv(T / "v3_signal_groups.csv"); G = G[G.metric == "pair_sec"].set_index("features")
V = pd.read_csv(T / "v1_past_only_called.csv"); V = V[V.metric == "pair_sec"].set_index("scorer")
rows = [("58 features + headline text", V.loc["hgb_bge"], INK), ("headline text only", V.loc["bge"], ORANGE),
        ("all 58 features", V.loc["hgb"], BLUE), ("history features only", G.loc["only_history"], AQUA),
        ("body features only", G.loc["only_body"], AQUA), ("title features only (5)", G.loc["only_title"], AQUA),
        ("section, weekday only", G.loc["only_placement"], AQUA), ("all but history", G.loc["without_history"], MUTED)]
yy = np.arange(len(rows))[::-1]
for yv, (lab, r, col) in zip(yy, rows):
    c.errorbar(100 * r.est, yv, xerr=[[100 * (r.est - r.lo)], [100 * (r.hi - r.est)]], fmt="o", ms=3.8, color=col, lw=1.3)
    c.text(64.3, yv, f"{100 * r.est:.1f}", fontsize=7, va="center", ha="right", color=INK)
c.axvline(50, color=MUTED, lw=0.8, ls="--"); c.set_yticks(yy, [r[0] for r in rows]); c.tick_params(axis="y", length=0)
c.set_xlabel("% correct, same day & section pairs"); c.set_xlim(49, 64.5)
c.spines[["top", "right"]].set_visible(False); c.set_title("(b) Which signals", loc="left", fontsize=8)
fig.tight_layout(w_pad=0.8); fig.savefig(ROOT / "results" / "figures" / "fig_called.png", dpi=300); print("saved")
