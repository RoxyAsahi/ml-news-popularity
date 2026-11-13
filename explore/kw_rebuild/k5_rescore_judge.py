"""Re-score the cached JEV judgements of v19 against the shares that really belong to the headlines it read.

v19 showed the judge real headlines taken from the url column and scored it against the shares in the same CSV row.
From 2013-06-06 on those shares belong to another article (k4_realign.py). No new API call is needed: each
headline's true row is known, so the cached answers are simply scored again.

(A) 4,000 same-day same-section pairs and 4,000 same-day pairs (explore/jev/results/api_mash_titles_*.csv):
    accuracy as scored in v19 and re-scored, by period; most-confident 25% / 10%; the past-only headline-word
    memory (k1, true alignment) and kw_avg_avg on the same re-scored pairs. After re-scoring the pair is still a
    same-day pair (a URL carries its own date); "same true section" marks pairs whose true rows share a section.
(B) Whole-day lineups (api_lineup.csv, the judge's probability that an article is the day's most shared, for every
    headline): mean within-day Spearman with log shares and how often its top pick lands in the day's top 10%,
    as scored in v19 and re-scored. Intervals: bootstrap over days.
Output: results/tables/k5_judge_rescored.csv; data/interim/kw_rebuild/judge_lineup_true.csv (file_row, p_top)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwlib import FIRST_SHIFTED, INT, ROOT, TAB, frame  # noqa: E402
from newspop.stories import channel  # noqa: E402

J = ROOT / "explore" / "jev" / "results"
B = 1000
rng = np.random.default_rng(0)
G, T = frame("given"), frame("true")
yG, yT = np.log1p(G.shares.to_numpy().astype(float)), np.log1p(T.shares.to_numpy().astype(float))
# true-frame row of the article whose URL sits in given-frame row i (NaN for the 152 URLs that belong to no row)
pos = pd.Series(np.arange(len(T)), index=T.url.to_numpy())
pos = pos[pos.index.notna()]
g2t = pos.reindex(G.url.to_numpy()).to_numpy()
chT = channel(T); dateT = T.publish_date.to_numpy(); dateG = G.publish_date.to_numpy()
FT = pd.read_parquet(INT / "features_true_L1.parquet")
mem = FT.tit_all_all_mean.to_numpy(); kwT = T.kw_avg_avg.to_numpy().astype(float)
rows = []


def boot(c, cl):
    u, inv = np.unique(cl, return_inverse=True); s, n = np.bincount(inv, c), np.bincount(inv)
    bs = [(w * s).sum() / (w * n).sum() for w in (np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B))]
    return float(np.mean(c)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def add(setting, period, scorer, scoring, c, cl):
    if len(c) < 50:
        return
    est, lo, hi = boot(np.asarray(c, float), cl)
    rows.append(dict(setting=setting, period=period, scorer=scorer, scoring=scoring, est=est, lo=lo, hi=hi, n=len(c)))


deb = lambda p0, p1: (np.asarray(p0) + 1 - np.asarray(p1)) / 2  # noqa: E731
PER = {"2013-01-07..2013-06-05": lambda d: d < FIRST_SHIFTED, "2013-06-06..2014-12-31": lambda d: d >= FIRST_SHIFTED,
       "all": lambda d: np.ones(len(d), bool)}
for mode in ["section", "day"]:
    f = J / f"api_mash_titles_{mode}.csv"
    if not f.exists():
        continue
    P = pd.read_csv(f); a, b = P.a.to_numpy(), P.b.to_numpy()
    p = deb(P.p0_title, P.p1_title) - .5; pk = deb(P.p0_slugt, P.p1_slugt) - .5
    ta, tb = g2t[a], g2t[b]; ok = ~np.isnan(ta) & ~np.isnan(tb)
    ta, tb = np.where(ok, ta, 0).astype(int), np.where(ok, tb, 0).astype(int)
    d = pd.to_datetime(dateG[a])
    setting = f"same-day same-{mode} pairs" if mode == "section" else "same-day pairs"
    for pname, fn in PER.items():
        sel = np.asarray(fn(d))
        # as scored in v19: shares of the CSV rows
        s1 = sel & (yG[a] != yG[b])
        add(setting, pname, "JEV, real headline", "as in v19 (url column as distributed)", ((p > 0) == (yG[a] > yG[b]))[s1], dateG[a][s1])
        # re-scored: shares of the rows the two headlines belong to
        s2 = sel & ok & (yT[ta] != yT[tb]); t = yT[ta] > yT[tb]
        add(setting, pname, "JEV, real headline", "re-scored (true rows)", ((p > 0) == t)[s2], dateG[a][s2])
        add(setting, pname, "JEV, URL keywords", "re-scored (true rows)", ((pk > 0) == t)[s2], dateG[a][s2])
        for q in (.25, .10):
            idx = np.flatnonzero(s2); top = idx[np.argsort(-np.abs(p[idx]), kind="stable")[: int(q * len(idx))]]
            add(setting, pname, f"JEV, real headline, most confident {int(q * 100)}%", "re-scored (true rows)", ((p > 0) == t)[top], dateG[a][top])
        dm = mem[ta] - mem[tb]; s3 = s2 & ~np.isnan(dm)
        add(setting, pname, "headline-word memory (past-only)", "re-scored (true rows)", ((dm > 0) == t)[s3] + .5 * (dm == 0)[s3], dateG[a][s3])
        dk = kwT[ta] - kwT[tb]
        add(setting, pname, "kw_avg_avg", "re-scored (true rows)", ((dk > 0) == t)[s2] + .5 * (dk == 0)[s2], dateG[a][s2])
        s4 = s2 & (chT[ta] == chT[tb])
        add(setting, pname, "JEV, real headline", "re-scored, same true section", ((p > 0) == t)[s4], dateG[a][s4])

# (B) lineups
f = J / "api_lineup.csv"
if f.exists():
    Lu = pd.read_csv(f)
    for scoring, yy, dd, r in [("as in v19 (url column as distributed)", yG[Lu.row], dateG[Lu.row], np.ones(len(Lu), bool)),
                               ("re-scored (true rows)", None, None, ~np.isnan(g2t[Lu.row]))]:
        if yy is None:
            tr = g2t[Lu.row][r].astype(int); yy, dd = yT[tr], dateT[tr]
            pd.DataFrame({"file_row": T.file_row.to_numpy()[tr], "p_top": Lu.p_top.to_numpy()[r]}).to_csv(INT / "judge_lineup_true.csv", index=False)
        t = pd.DataFrame({"p": Lu.p_top.to_numpy()[r], "y": yy, "d": dd})
        for pname, fn in PER.items():
            rho, top, base, days = [], [], [], []
            for dday, g in t[np.asarray(fn(pd.to_datetime(t.d)))].groupby("d"):
                if len(g) < 10:
                    continue
                rho.append(np.corrcoef(rankdata(g.p), rankdata(g.y))[0, 1]); thr = np.quantile(g.y, .9)
                top.append(float(g.y.to_numpy()[np.argmax(g.p.to_numpy())] >= thr)); base.append(np.mean(g.y >= thr)); days.append(dday)
            add("whole-day lineup", pname, "JEV, within-day Spearman", scoring, rho, np.arange(len(rho)))
            add("whole-day lineup", pname, "JEV, top pick in day's top 10%", scoring, top, np.arange(len(top)))
            add("whole-day lineup", pname, "random pick in day's top 10%", scoring, base, np.arange(len(base)))

R = pd.DataFrame(rows); R.to_csv(TAB / "k5_judge_rescored.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R.round(3).to_string(index=False))
