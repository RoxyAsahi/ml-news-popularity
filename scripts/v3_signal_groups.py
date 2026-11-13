"""v19 RQ2: which kind of signal carries what can be called before publication?

Past-only HistGB (monthly rolling origin, as in v1_past_only_called.py) on four feature groups alone and on all
features without each group:
  history    keyword and self-reference statistics (shares of earlier articles: kw_*, self_reference_*)
  placement  channel and weekday flags
  body       length, links, media, LDA topics, sentiment of the body
  title      the dataset's title features (length, subjectivity, polarity)
Metrics on 2013-07 .. 2014-12: AUC (popular = shares >= 1400), mean within-day Spearman, accuracy on all
same-day same-section pairs; 95% intervals from 1,000 bootstrap draws of days.
Output: results/tables/v3_signal_groups.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.stories import channel  # noqa: E402

FIRST, B = "2013-07", 1000
df = load().reset_index(drop=True)
y = np.log1p(df.shares.to_numpy()); pop = (df.shares.to_numpy() >= 1400).astype(int)
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
cols = list(X.columns)
G = {"history": [c for c in cols if c.startswith(("kw_", "self_reference_"))],
     "placement": [c for c in cols if c.startswith(("data_channel_", "weekday_", "is_weekend"))],
     "title": [c for c in cols if "title" in c]}
G["body"] = [c for c in cols if not any(c in v for v in G.values())]
for k, v in G.items():
    print(k, len(v), flush=True)
SETS = {"all": cols, **{f"only_{k}": v for k, v in G.items()}, **{f"without_{k}": [c for c in cols if c not in v] for k, v in G.items()}}

day = pd.to_datetime(df.publish_date).dt.normalize(); dn = (day - day.min()).dt.days.to_numpy()
month = day.dt.to_period("M").astype(str).to_numpy(); chan = channel(df)
ev = month >= FIRST
scores = {k: np.full(len(df), np.nan) for k in SETS}
for m in sorted(set(month)):
    if m < FIRST:
        continue
    tr, te = month < m, month == m
    for k, c in SETS.items():
        scores[k][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(X.loc[tr, c], y[tr]).predict(X.loc[te, c])
    print(m, flush=True)

idx_ev = np.flatnonzero(ev)
days = np.unique(dn[ev]); D = len(days)
day_rows = [idx_ev[dn[idx_ev] == d] for d in days]
di = np.searchsorted(days, dn[idx_ev])


def per_day(s):
    rho, cs, ts = np.full(D, np.nan), np.zeros(D), np.zeros(D)
    for d, idx in enumerate(day_rows):
        if len(idx) >= 10:
            rho[d] = np.corrcoef(rankdata(s[idx]), rankdata(y[idx]))[0, 1]
        for c in np.unique(chan[idx]):
            g = idx[chan[idx] == c]
            if len(g) < 2:
                continue
            a, b = np.triu_indices(len(g), 1)
            dy = np.sign(y[g[a]] - y[g[b]]); ds = np.sign(s[g[a]] - s[g[b]]); k = dy != 0
            cs[d] += np.sum((dy[k] == ds[k]) + 0.5 * (ds[k] == 0)); ts[d] += k.sum()
    return rho, cs, ts


rng = np.random.default_rng(0)
W = [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(B)]
rows = []
for k in SETS:
    s = scores[k]; rho, cs, ts = per_day(s); ok = ~np.isnan(rho)

    def stat(w):
        return (roc_auc_score(pop[idx_ev], s[idx_ev], sample_weight=w[di]),
                np.sum(w[ok] * rho[ok]) / np.sum(w[ok]), np.sum(w * cs) / np.sum(w * ts))
    est = stat(np.ones(D)); bs = np.array([stat(w) for w in W])
    for j, mname in enumerate(["auc", "rho_day", "pair_sec"]):
        rows.append(dict(features=k, n_features=len(SETS[k]), metric=mname, est=est[j],
                         lo=np.quantile(bs[:, j], .025), hi=np.quantile(bs[:, j], .975)))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / "v3_signal_groups.csv", index=False)
print(R.pivot(index="features", columns="metric", values="est").round(4))
