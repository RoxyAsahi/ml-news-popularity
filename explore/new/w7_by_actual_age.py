"""Direction A: the same comparison indexed by the ACTUAL age of the observation, not by a nominal horizon.

w2-w6 use "the last look within h hours", whose real age is much younger than h (median 3.8 h for h = 6, 8.5 h for
h = 24) because the archive holds only a few captures a day. Here an article enters an age band if it has a sighting
whose age falls inside the band (the last one inside the band is used; nothing later is seen).
Past-only monthly rolling origin. Scorers: Szabo-Huberman (log final = a + log count), prepub (58 features),
early (counts per network + state + velocity), both. AUC for >= 1400 on all articles of the band and on those not
yet past 1,400, AUC for the top 5%, Spearman.
Output: results/tables/w7_by_actual_age.csv
"""
from __future__ import annotations

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

FIRST = "2013-04"
BANDS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
df = load_fixed()
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float); y = np.log1p(s)
month = df.date.dt.to_period("M").astype(str).to_numpy()
t95 = np.quantile(s, .95)
O = load_obs(df, lists_only=True)
rows = []
for lo, hi in BANDS:
    E = early_state(O[O.age_h < hi], hi)  # history up to the band's end
    E = E[E.age >= lo]                    # keep articles whose last sighting lies inside the band
    idx = E.index.to_numpy(); mi = month[idx]; yi = y[idx]; si = s[idx]; lc = E.l_total.to_numpy()
    A = E[COUNT_COLS + STATE_COLS + VEL_COLS].to_numpy(dtype=float); A[:, np.isnan(A).all(0)] = 0.0
    mats = {"prepub": X[idx], "early": A, "both": np.c_[X[idx], A]}
    sc = {k: np.full(len(idx), np.nan) for k in mats}
    for m in sorted(set(mi)):
        tr, te = mi < m, mi == m
        if m < FIRST or tr.sum() < 300 or te.sum() == 0:
            continue
        for k, M in mats.items():
            sc[k][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(M[tr], yi[tr]).predict(M[te])
    ok = ~np.isnan(sc["prepub"]); sc["szabo_huberman"] = lc
    lab = si[ok] >= 1400; below = np.expm1(lc[ok]) < 1400
    for k in ["szabo_huberman", "prepub", "early", "both"]:
        z = sc[k][ok]
        rows.append(dict(age_band=f"{lo}-{hi} h", n=int(ok.sum()), age_median=float(np.median(E.age.to_numpy()[ok])),
                         already_1400=float((~below).mean()), scorer=k, auc1400=roc_auc_score(lab, z),
                         auc_not_yet=roc_auc_score(lab[below], z[below]), auc_top5=roc_auc_score(si[ok] >= t95, z),
                         spearman=spearmanr(z, yi[ok])[0]))
    print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "w7_by_actual_age.csv", index=False)
