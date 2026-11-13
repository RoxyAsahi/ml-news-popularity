"""What the shifted url column did to every headline-based result, and what the headline carries once it is fixed.

Signals (all strictly past-only, lag 1 day, from k1_build_features.py), under both alignments of kwlib.frame:
  headline-word memory   tit_all_all_mean   EB lift of the headline's words, all history
  slug-word memory       slug_all_all_mean
  kw_avg_avg             dataset column (for reference; a row-wise column, so only its grouping by day changes)
(A) 14-day series of the mean within-(day x section) Spearman with log shares, 95% intervals over days.
(B) Under the alignment as given: least-squares single change point of the daily within-(day x section) correlation
    (weighted by articles), with a day bootstrap for the date.
(C) Periods: before the first shifted row (2013-03-01 .. 2013-06-05), 2013-06-06 .. 2013-12-31, 2014.
(D) Split-half reliability of the lifts of frequent headline words (random halves of the articles, 200 splits) per
    period: do words carry a stable popularity difference inside a day and section at all?
(E) Which memory horizon carries the signal (true alignment): all history, windows of 7/30/90 days, exponential
    decay, raw-share mimic of the original recipe, momentum; and lags 1 .. 30 days. Reported on 2013-03..06 (before
    the evaluation window, where the variant is chosen) and on the evaluation window 2013-07 .. 2014-12.
Output: results/tables/k3_series.csv, k3_change_point.csv, k3_periods.csv, k3_word_reliability.csv,
        k3_horizons.csv; results/figures/k3_headline_signal.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k1_build_features as k1  # noqa: E402
from kwlib import FIG, FIRST_SHIFTED, INT, TAB, frame  # noqa: E402
from newspop.stories import channel  # noqa: E402

B = 1000
rng = np.random.default_rng(0)
D0 = pd.Timestamp("2013-01-07")


class View:
    def __init__(self, align):
        self.df = df = frame(align)
        self.y = np.log1p(df.shares.to_numpy().astype(float))
        self.date = df.publish_date
        self.dn = (df.publish_date - D0).dt.days.to_numpy()
        self.grp = self.dn * 10 + pd.factorize(channel(df))[0]
        self.F = {L: pd.read_parquet(INT / f"features_{align}_L{L}.parquet") for L in [1, 3, 7, 14, 30]}
        self.nd = self.dn.max() + 1

    def daily(self, v):
        """Per day: sum over (day x section) groups of n_g * Spearman(v, y), and sum of n_g."""
        num, den = np.zeros(self.nd), np.zeros(self.nd)
        t = pd.DataFrame({"v": v, "y": self.y, "g": self.grp, "d": self.dn}).dropna()
        for _, x in t.groupby("g"):
            if len(x) >= 5 and x.v.nunique() > 1 and x.y.nunique() > 1:
                num[x.d.iloc[0]] += len(x) * np.corrcoef(rankdata(x.v), rankdata(x.y))[0, 1]; den[x.d.iloc[0]] += len(x)
        return num, den

    def days(self, lo, hi):
        return np.arange(max((pd.Timestamp(lo) - D0).days, 0), min((pd.Timestamp(hi) - D0).days + 1, self.nd))


def ratio_ci(num, den, days):
    est = num[days].sum() / max(den[days].sum(), 1e-9)
    bs = [num[s].sum() / max(den[s].sum(), 1e-9) for s in (days[rng.integers(0, len(days), len(days))] for _ in range(B))]
    return est, np.quantile(bs, .025), np.quantile(bs, .975)


V = {a: View(a) for a in ["given", "true"]}
SIG = {"headline-word memory": "tit_all_all_mean", "slug-word memory": "slug_all_all_mean"}
DW = {}
for a, v in V.items():
    for name, col in SIG.items():
        DW[(a, name)] = v.daily(v.F[1][col].to_numpy())
    DW[(a, "kw_avg_avg")] = v.daily(v.df.kw_avg_avg.to_numpy().astype(float))

# (A) series, bins anchored on the first shifted day
rows = []
b0 = (FIRST_SHIFTED - D0).days % 14
for (a, name), (num, den) in DW.items():
    for s in range(b0, V[a].nd, 14):
        days = np.arange(s, min(s + 14, V[a].nd))
        est, lo, hi = ratio_ci(num, den, days)
        rows.append(dict(align=a, signal=name, start=(D0 + pd.Timedelta(days=int(s))).date(), rho=est, lo=lo, hi=hi))
A = pd.DataFrame(rows); A.to_csv(TAB / "k3_series.csv", index=False)


# (B) change point under the alignment as given
def change_point(num, den, days):
    days = np.sort(days); w = den[days]; r = np.where(w > 0, num[days] / np.maximum(w, 1e-9), 0.0)
    cw, cwr, cwr2 = np.cumsum(w), np.cumsum(w * r), np.cumsum(w * r * r)
    i = np.arange(1, len(days)); ok = (cw[i - 1] > 0) & (cw[-1] - cw[i - 1] > 0) & (days[i] > 40) & (days[i] < days.max() - 40)
    m1 = cwr[i - 1] / np.maximum(cw[i - 1], 1e-9); m2 = (cwr[-1] - cwr[i - 1]) / np.maximum(cw[-1] - cw[i - 1], 1e-9)
    sse = np.where(ok, cwr2[-1] - cw[i - 1] * m1 ** 2 - (cw[-1] - cw[i - 1]) * m2 ** 2, np.inf)
    j = int(np.argmin(sse))
    return days[i[j]], m1[j], m2[j], 1 - sse[j] / (cwr2[-1] - cwr[-1] ** 2 / cw[-1])


rows = []
for (a, name), (num, den) in DW.items():
    nd = V[a].nd
    c, m1, m2, r2 = change_point(num, den, np.arange(nd))
    cs = [change_point(num, den, rng.integers(0, nd, nd))[0] for _ in range(300)]
    d = lambda x: (D0 + pd.Timedelta(days=int(x))).date()  # noqa: E731
    rows.append(dict(align=a, signal=name, change_point=d(c), level_before=m1, level_after=m2, variance_explained=r2,
                     date_lo=d(np.quantile(cs, .025)), date_hi=d(np.quantile(cs, .975))))
Bk = pd.DataFrame(rows); Bk.to_csv(TAB / "k3_change_point.csv", index=False)

# (C) periods
PER = {"2013-03-01..2013-06-05 (file aligned)": ("2013-03-01", "2013-06-05"), "2013-06-06..2013-12-31": ("2013-06-06", "2013-12-31"),
       "2014": ("2014-01-01", "2014-12-31"), "2013-07-01..2014-12-31 (v19 window)": ("2013-07-01", "2014-12-31")}
rows = []
for (a, name), (num, den) in DW.items():
    for p, (lo, hi) in PER.items():
        est, l, h = ratio_ci(num, den, V[a].days(lo, hi))
        rows.append(dict(align=a, signal=name, period=p, rho=est, lo=l, hi=h))
C = pd.DataFrame(rows); C.to_csv(TAB / "k3_periods.csv", index=False)

# (D) split-half reliability of word lifts
WP = {"2013-01-07..2013-06-05": ("2013-01-07", "2013-06-05"), "2013-06-06..2013-12-31": ("2013-06-06", "2013-12-31"),
      "2014-01..06": ("2014-01-01", "2014-06-30"), "2014-07..12": ("2014-07-01", "2014-12-31")}
rows = []
for a, v in V.items():
    keys = k1.keys_for(v.df)["tit"]
    r = v.y - pd.Series(v.y).groupby(v.grp).transform("mean").to_numpy()
    L = pd.DataFrame({"art": np.repeat(np.arange(len(keys)), [len(x) for x in keys]), "key": [w for x in keys for w in x]})
    L["r"] = r[L.art]
    for p, (lo, hi) in WP.items():
        m = ((v.date >= lo) & (v.date <= hi)).to_numpy()
        Lp = L[m[L.art]]
        cnt = Lp.key.value_counts(); Lp = Lp[Lp.key.isin(cnt.index[cnt >= 40])]
        arts = Lp.art.unique(); rs = []
        for _ in range(200):
            h = Lp.art.isin(set(rng.permutation(arts)[: len(arts) // 2]))
            g1, g2 = Lp[h].groupby("key").r.mean(), Lp[~h].groupby("key").r.mean()
            j = g1.index.intersection(g2.index); rs.append(np.corrcoef(g1[j], g2[j])[0, 1])
        rows.append(dict(align=a, period=p, words=int((cnt >= 40).sum()), split_half_r=float(np.mean(rs)),
                         lo=float(np.quantile(rs, .025)), hi=float(np.quantile(rs, .975))))
Dd = pd.DataFrame(rows); Dd.to_csv(TAB / "k3_word_reliability.csv", index=False)

# (E) horizons and lags, true alignment, v19 window
v = V["true"]
rows = []
H = {"all history, EB lift": "all_all_mean", "all history, raw shares (original recipe)": "mim_avg_avg", "window 90 d": "win_w90_mean",
     "window 30 d": "win_w30_mean", "window 7 d": "win_w7_mean", "decay, half-life 90 d": "dec_h90_mean", "decay, half-life 30 d": "dec_h30_mean",
     "decay, half-life 7 d": "dec_h7_mean", "momentum 7 d minus all": "mom_w7_minus_all_mean", "momentum 30 d minus all": "mom_w30_minus_all_mean",
     "all history, EB lift, max over words": "all_all_max", "all history, EB lift, sum over words": "all_all_sum"}
EP = {"2013-03-01..2013-06-30 (before the evaluation window; variant chosen here)": ("2013-03-01", "2013-06-30"),
      "2013-07-01..2014-12-31 (evaluation window)": ("2013-07-01", "2014-12-31")}
for src in ["tit", "slug"]:
    for name, col in H.items():
        nd = v.daily(v.F[1][f"{src}_{col}"].to_numpy())
        for p, (a, b) in EP.items():
            est, lo, hi = ratio_ci(*nd, v.days(a, b))
            rows.append(dict(period=p, source=src, feature=name, lag_days=1, rho=est, lo=lo, hi=hi))
    for L in [3, 7, 14, 30]:
        nd = v.daily(v.F[L][f"{src}_all_all_mean"].to_numpy())
        for p, (a, b) in EP.items():
            est, lo, hi = ratio_ci(*nd, v.days(a, b))
            rows.append(dict(period=p, source=src, feature="all history, EB lift", lag_days=L, rho=est, lo=lo, hi=hi))
for p, (a, b) in EP.items():
    est, lo, hi = ratio_ci(*DW[("true", "kw_avg_avg")], v.days(a, b))
    rows.append(dict(period=p, source="dataset", feature="kw_avg_avg", lag_days=np.nan, rho=est, lo=lo, hi=hi))
E = pd.DataFrame(rows); E.to_csv(TAB / "k3_horizons.csv", index=False)

# figure
fig, ax = plt.subplots(figsize=(7.2, 3.3))
style = {("given", "headline-word memory"): ("#c0392b", "--", "headline-word memory, url column as distributed"),
         ("true", "headline-word memory"): ("#c0392b", "-", "headline-word memory, url column re-aligned"),
         ("true", "kw_avg_avg"): ("#2c3e50", "-", "kw_avg_avg (dataset column)")}
for (a, name), (col, ls, lab) in style.items():
    g = A[(A["align"] == a) & (A.signal == name)]; x = pd.to_datetime(g.start) + pd.Timedelta(days=7)
    ax.plot(x, g.rho, color=col, ls=ls, lw=1.5, label=lab); ax.fill_between(x, g.lo, g.hi, color=col, alpha=.12, lw=0)
ax.axvline(FIRST_SHIFTED, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=.5)
ax.text(FIRST_SHIFTED + pd.Timedelta(days=6), -0.07, "first shifted row\n2013-06-06", fontsize=7)
ax.set_ylabel("Spearman with log shares\nwithin day and section"); ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=1)
ax.set_ylim(-0.1, 0.56)
fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "k3_headline_signal.png", dpi=200)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for t in (Bk, C, Dd, E):
    print(t.round(3).to_string(index=False), "\n")
