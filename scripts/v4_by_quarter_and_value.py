"""v19 checks asked for in review round 11.
(A) Accuracy by publication quarter on the 4,000 same-day same-section pairs judged by the API: the judge with real
    headlines and with URL keywords only, kw_avg_avg (no fitting), HistGB with random 5-fold cross-validation
    (explore/jev/results/mash_oof.csv, trained on all periods) and past-only HistGB (from 2013-07). If the scorers
    that cannot remember articles are also better in early 2013, the period is easier; if only the judge is, the
    judge may know older stories. 95% intervals over days.
(B) Decision value of the past-only top pick: per day, shares of the top-scored article vs the day's median article
    and vs a random article (geometric mean ratio over days, day bootstrap).
Output: results/tables/v4_by_quarter.csv, results/tables/v4_top_pick_value.csv, results/figures/v19_fig_quarter.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402
J = ROOT / "explore" / "jev" / "results"; T = ROOT / "results" / "tables"
rng = np.random.default_rng(0); B = 1000
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy())
day = pd.to_datetime(df.publish_date).dt.normalize().to_numpy()
quarter = pd.to_datetime(df.publish_date).dt.to_period("Q").astype(str).to_numpy()
P = pd.read_csv(J / "api_mash_titles_section.csv"); a, b = P.a.to_numpy(), P.b.to_numpy(); t = y[a] > y[b]
O = pd.read_csv(J / "mash_oof.csv")
S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv"); hp = np.full(len(df), np.nan); hp[S.row] = S.hgb
deb = lambda p0, p1: (np.asarray(p0) + 1 - np.asarray(p1)) / 2
sc = {"JEV judge, headline": deb(P.p0_title, P.p1_title) - .5,
      "JEV judge, URL keywords": deb(P.p0_slugt, P.p1_slugt) - .5,
      "kw_avg_avg": df.kw_avg_avg.to_numpy()[a] - df.kw_avg_avg.to_numpy()[b],
      "HistGB, random 5-fold": O.hgb.to_numpy()[a] - O.hgb.to_numpy()[b],
      "HistGB, past-only": hp[a] - hp[b]}


def boot(c, cl):
    u, inv = np.unique(cl, return_inverse=True); s, n = np.bincount(inv, c), np.bincount(inv)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)); bs.append((w * s).sum() / (w * n).sum())
    return c.mean(), np.quantile(bs, .025), np.quantile(bs, .975)


rows = []
for name, d in sc.items():
    for q in sorted(set(quarter[a])):
        sel = (quarter[a] == q) & ~np.isnan(d)
        if sel.sum() < 100:
            continue
        c = ((d[sel] > 0) == t[sel]).astype(float) + 0.5 * (d[sel] == 0) * 0
        est, lo, hi = boot(c, day[a][sel])
        rows.append(dict(scorer=name, quarter=q, acc=est, lo=lo, hi=hi, n=int(sel.sum())))
Q = pd.DataFrame(rows); Q.to_csv(T / "v4_by_quarter.csv", index=False)
print(Q.pivot(index="quarter", columns="scorer", values="acc").round(3).to_string())
# correlation over quarters between judge and other scorers' accuracy
piv = Q.pivot(index="quarter", columns="scorer", values="acc")
print("corr over quarters with JEV headline:\n", piv.corr()["JEV judge, headline"].round(2).to_string())

# (B) top-pick value
S = S[S.has_title]; S["d"] = day[S.row]; S["shares"] = df.shares.to_numpy()[S.row]
lift_med, lift_rand, days = [], [], []
for d_, g in S.groupby("d"):
    if len(g) < 10:
        continue
    top = g.shares.to_numpy()[np.argmax(g.hgb.to_numpy())]
    lift_med.append(np.log(top) - np.log(np.median(g.shares))); lift_rand.append(np.log(top) - np.mean(np.log(g.shares)))
    days.append(top)
lm, lr = np.array(lift_med), np.array(lift_rand)
bs = [(np.exp(lm[i].mean()), np.exp(lr[i].mean())) for i in [rng.integers(0, len(lm), len(lm)) for _ in range(B)]]
bs = np.array(bs)
V = pd.DataFrame([dict(metric="geo_mean_ratio_top_vs_day_median", est=np.exp(lm.mean()), lo=np.quantile(bs[:, 0], .025), hi=np.quantile(bs[:, 0], .975)),
                  dict(metric="geo_mean_ratio_top_vs_random_article", est=np.exp(lr.mean()), lo=np.quantile(bs[:, 1], .025), hi=np.quantile(bs[:, 1], .975)),
                  dict(metric="median_shares_top_pick", est=np.median(days), lo=np.nan, hi=np.nan),
                  dict(metric="median_shares_all_scored", est=np.median(S.shares), lo=np.nan, hi=np.nan),
                  dict(metric="share_top_pick_above_day_median", est=np.mean(lm > 0), lo=np.nan, hi=np.nan),
                  dict(metric="days", est=len(lm), lo=np.nan, hi=np.nan)])
V.to_csv(T / "v4_top_pick_value.csv", index=False); print(V.round(3).to_string(index=False))

# figure: accuracy by quarter
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
style = {"JEV judge, headline": (ORANGE, "-", "o"), "JEV judge, URL keywords": (ORANGE, (0, (2, 1.5)), "s"),
         "kw_avg_avg": (AQUA, "-", "o"), "HistGB, random 5-fold": (BLUE, (0, (2, 1.5)), "s"), "HistGB, past-only": (BLUE, "-", "o")}
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
qs = sorted(set(Q.quarter)); xi = {q: i for i, q in enumerate(qs)}
for name, (col, ls, mk) in style.items():
    g = Q[Q.scorer == name]; x = np.array([xi[q] for q in g.quarter]) + {"o": 0.06, "s": -0.06}[mk]
    ax.plot(x, 100 * g.acc, color=col, ls=ls, marker=mk, ms=3, lw=1.4, label=name)
    if name in ("JEV judge, headline", "HistGB, past-only"):
        ax.fill_between(x, 100 * g.lo, 100 * g.hi, color=col, alpha=0.12, lw=0)
ax.axhline(50, color=MUTED, lw=0.8, ls="--")
ax.set_xticks(range(len(qs)), [q.replace("20", "'", 1).replace("Q", " Q") for q in qs], fontsize=6.5)
ax.set_ylabel("% correct, same day & section pairs"); ax.set_ylim(40, 78)
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", columnspacing=0.8, handlelength=2.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "v19_fig_quarter.png", dpi=300); print("saved")
