"""Decision value of the past-only top pick (repaired data).

Per day in the scored months (at least 10 articles), the article a scorer ranks first is compared with the day's
median article and with a random article of that day: geometric-mean ratio of shares over days, share of days on
which the pick beats the day's median, and the median shares of the picks. Scorers: past-only HistGB on the 58
features, the same plus the headline embedding, the headline model alone, and kw_avg_avg without fitting.
95% intervals: bootstrap over days (1,000 draws), paired across scorers.
Input: data/interim/v1_scores.csv (scripts/v1_past_only_called.py). Output: results/tables/v4_top_pick_value.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402
rng = np.random.default_rng(0); B = 1000
df = load().reset_index(drop=True)
S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv"); S = S[S.has_title].copy()
S["shares"] = df.shares.to_numpy()[S.row]
SCORERS = ["hgb", "hgb_bge", "bge", "tfidf", "kw_avg_avg"]
G = [g for _, g in S.groupby("day") if len(g) >= 10]
draws = [rng.integers(0, len(G), len(G)) for _ in range(B)]
rows = []
L = {}
for s in SCORERS:
    top = np.array([g.shares.to_numpy()[np.argmax(g[s].to_numpy())] for g in G], float)
    med = np.array([np.median(g.shares) for g in G]); gm = np.array([np.mean(np.log(g.shares)) for g in G])
    lm, lr = np.log(top) - np.log(med), np.log(top) - gm; L[s] = lr
    for metric, x, f in [("geo_mean_ratio_top_vs_day_median", lm, np.exp), ("geo_mean_ratio_top_vs_random_article", lr, np.exp),
                         ("share_days_top_pick_above_day_median", (lm > 0).astype(float), lambda v: v)]:
        bs = [f(x[i].mean()) for i in draws]
        rows.append(dict(scorer=s, metric=metric, est=f(x.mean()), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    rows.append(dict(scorer=s, metric="median_shares_top_pick", est=np.median(top), lo=np.nan, hi=np.nan))
d = L["hgb_bge"] - L["hgb"]; bs = [np.exp(d[i].mean()) for i in draws]
rows.append(dict(scorer="hgb_bge_vs_hgb", metric="ratio_of_geo_mean_ratios", est=np.exp(d.mean()), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
rows.append(dict(scorer="all", metric="median_shares_all_scored", est=np.median(S.shares), lo=np.nan, hi=np.nan))
rows.append(dict(scorer="all", metric="days", est=len(G), lo=np.nan, hi=np.nan))
V = pd.DataFrame(rows); V.to_csv(ROOT / "results" / "tables" / "v4_top_pick_value.csv", index=False)
print(V.round(3).to_string(index=False))
