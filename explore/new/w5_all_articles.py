"""Direction A, selection check: the same question over ALL dataset articles, not only the ones the archive saw.

w2-w4 score only articles with a sighting at age <= h, and which articles get sighted depends on how they are
doing (winners stay on the homepage longer). Here every article of the months covered by the captures is scored.
The information at h hours is "whatever the archive shows by then": the early state if the article was sighted,
otherwise missing values (the model may learn from absence too), plus the number of homepage captures taken on the
publication day and the next day (so that "nobody looked" differs from "looked and it was not there").
This is a lower bound on what a newsroom dashboard, which sees every article all the time, would allow.

Past-only monthly rolling origin. Scorers: prepub (58 features), both (58 + archive state at h).
Metrics over all articles and split by sighted / not sighted; day-bootstrap interval for the AUC difference.
Output: results/tables/w5_all_articles.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COUNT_COLS, HGB_PARAMS, ROOT, STATE_COLS, VEL_COLS, early_state, features, load_fixed,  # noqa: E402
                    load_obs)

ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2013-04")
ap.add_argument("--horizons", default="1,3,6,12,24")
ap.add_argument("--boot", type=int, default=300)
a = ap.parse_args()

df = load_fixed()
n = len(df)
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float)
y = np.log1p(s)
month = df.date.dt.to_period("M").astype(str).to_numpy()
dn = (df.date - df.date.min()).dt.days.to_numpy()
t95 = np.quantile(s, .95)
O = load_obs(df, lists_only=True)
raw = pd.read_parquet(ROOT / "data" / "interim" / "wb_obs.parquet", columns=["ts"])
cap_day = pd.Series(pd.to_datetime(raw.ts.unique())).dt.normalize().value_counts()
ncap = (df.date.map(cap_day).fillna(0) + (df.date + pd.Timedelta(days=1)).map(cap_day).fillna(0)).to_numpy()
last_month = O.ts.max().strftime("%Y-%m")
use = month < last_month  # full months only
cols = COUNT_COLS + STATE_COLS + VEL_COLS
rng = np.random.default_rng(0)
rows = []
for h in [float(x) for x in a.horizons.split(",")]:
    E = early_state(O, h).reindex(np.arange(n))
    A = np.c_[X, E[cols].to_numpy(dtype=float), E.age.notna().to_numpy(dtype=float), ncap]
    seen = E.age.notna().to_numpy()
    sc = {"prepub": np.full(n, np.nan), "both": np.full(n, np.nan)}
    for m in sorted(set(month[use])):
        tr, te = use & (month < m), use & (month == m)
        if m < a.first or tr.sum() < 1000:
            continue
        ok_cols = ~np.isnan(A[tr]).all(0)
        sc["prepub"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(X[tr], y[tr]).predict(X[te])
        sc["both"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(A[tr][:, ok_cols], y[tr]).predict(A[te][:, ok_cols])
    ev = ~np.isnan(sc["prepub"])
    days = np.unique(dn[ev]); gi = [np.flatnonzero(dn[ev] == d) for d in days]
    boot = rng.integers(0, len(days), size=(a.boot, len(days)))
    lab = s[ev] >= 1400; zp, zb = sc["prepub"][ev], sc["both"][ev]
    d = [roc_auc_score(lab[sub], zb[sub]) - roc_auc_score(lab[sub], zp[sub]) for sub in (np.concatenate([gi[j] for j in b]) for b in boot)]
    for grp, mask in [("all", np.ones(ev.sum(), bool)), ("sighted", seen[ev]), ("not_sighted", ~seen[ev])]:
        r = dict(h=h, group=grp, n=int(mask.sum()), share=float(mask.mean()), p1400=float(lab[mask].mean()))
        for k, z in [("prepub", zp), ("both", zb)]:
            r["auc1400_" + k] = roc_auc_score(lab[mask], z[mask])
            r["auc_top5_" + k] = roc_auc_score(s[ev][mask] >= t95, z[mask])
            r["spearman_" + k] = spearmanr(z[mask], y[ev][mask])[0]
        if grp == "all":
            r["d_auc1400_lo"], r["d_auc1400_hi"] = np.quantile(d, [.025, .975])
        rows.append(r)
    print(pd.DataFrame(rows[-3:]).round(3).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "w5_all_articles.csv", index=False)
