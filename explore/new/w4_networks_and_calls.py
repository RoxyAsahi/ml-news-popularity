"""Direction A, deep dive 2: which network carries the signal, and how early can a blockbuster be called?

Look at an article h hours after publication (its last archive sighting at age <= h). All models are HGB
regressors of log final shares, past-only (monthly rolling origin, fit on months < m, score month m).

Part A  one network at a time: age + the count on that network only (twitter / facebook / linked_in / google_plus),
        against age + total and age + all networks.
Part B  calling blockbusters (final shares above the 95th percentile of all earlier months), with
          prepub      the 58 dataset features
          early       age + counts per network + Mashable state and velocity
          both        everything
        - precision of the top 5% of scores in each month, and recall of blockbusters among them;
        - conformal selection (Jin & Candes 2023) at level q: calibration = scored articles of earlier months,
          Benjamini-Hochberg on the month's conformal p-values; reports shortlist size, realised false-discovery
          proportion and power, and how many shortlisted articles had ALREADY passed the bar at the look.

Outputs: results/tables/w4_networks.csv, results/tables/w4_calls.csv, data/interim/w4_scores.parquet
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
ap.add_argument("--horizons", default="1,2,3,6,12,24")
a = ap.parse_args()
OUT = ROOT / "results" / "tables"

df = load_fixed()
n = len(df)
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float)
y = np.log1p(s)
month = df.date.dt.to_period("M").astype(str).to_numpy()
t95 = np.quantile(s, .95)
O = load_obs(df, lists_only=True)
print(f"sightings {len(O):,} on {O.row.nunique():,} dataset articles, captures up to {O.ts.max()}", flush=True)


def reg():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


def fill_dead(A):
    A = A.copy(); A[:, np.isnan(A).all(0)] = 0.0
    return A


SETS = {"twitter": ["age", "l_twitter"], "facebook": ["age", "l_facebook"], "linked_in": ["age", "l_linked_in"],
        "google_plus": ["age", "l_google_plus"], "total": ["age", "l_total"], "all_networks": COUNT_COLS,
        "early": COUNT_COLS + STATE_COLS + VEL_COLS}
nrows, crows, keep = [], [], []
for h in [float(x) for x in a.horizons.split(",")]:
    E = early_state(O, h)
    idx = E.index.to_numpy(); mi = month[idx]; yi = y[idx]; si = s[idx]
    mats = {k: fill_dead(E[c].to_numpy(dtype=float)) for k, c in SETS.items()}
    mats["prepub"] = X[idx]
    mats["both"] = np.c_[X[idx], mats["early"]]
    sc = {k: np.full(len(idx), np.nan) for k in mats}
    for m in sorted(set(mi)):
        tr, te = mi < m, mi == m
        if m < a.first or tr.sum() < 300 or te.sum() == 0:
            continue
        for k, A in mats.items():
            sc[k][te] = reg().fit(A[tr], yi[tr]).predict(A[te])
    ok = ~np.isnan(sc["prepub"])
    cur = np.expm1(E.l_total.to_numpy())
    for k in sc:
        z = sc[k][ok]
        nrows.append(dict(h=h, features=k, n=int(ok.sum()), auc1400=roc_auc_score(si[ok] >= 1400, z),
                          auc_top5=roc_auc_score(si[ok] >= t95, z), spearman=spearmanr(z, yi[ok])[0]))
    print(pd.DataFrame(nrows[-len(sc):]).round(3).to_string(index=False), flush=True)
    keep.append(pd.DataFrame({"h": h, "row": idx[ok], "cur": cur[ok], **{k: sc[k][ok] for k in ["prepub", "early", "both"]}}))
    # ---------------- Part B: calling blockbusters
    ms = sorted(set(mi[ok]))
    for k in ["prepub", "early", "both"]:
        v = sc[k]
        tp = sel = pos = 0
        for m in ms:
            te = ok & (mi == m); c = np.quantile(s[month < m], .95)
            kk = max(1, int(round(.05 * te.sum())))
            top = np.flatnonzero(te)[np.argsort(-v[te])[:kk]]
            tp += (si[top] > c).sum(); sel += kk; pos += (si[te] > c).sum()
        base = dict(h=h, scorer=k, n=int(ok.sum()), base_rate=pos / ok.sum(), top5pct_precision=tp / sel, top5pct_recall=tp / pos)
        for q in (0.1, 0.2, 0.3, 0.5):
            R = D = T = A0 = 0
            for m in ms[2:]:
                cal = ok & (mi < m); te = ok & (mi == m); c = np.quantile(s[month < m], .95)
                vn = np.sort(v[cal & (si <= c)])
                p = (1 + (len(vn) - np.searchsorted(vn, v[te], side="left"))) / (cal.sum() + 1)
                o = np.argsort(p); r_ = np.arange(1, te.sum() + 1)
                good = np.flatnonzero(p[o] <= q * r_ / te.sum())
                r = good.max() + 1 if len(good) else 0
                pick = np.flatnonzero(te)[o[:r]]
                R += r; D += (si[pick] > c).sum(); T += (si[te] > c).sum(); A0 += (cur[pick] > c).sum()
            crows.append(dict(**base, q=q, shortlisted=R, fdp=1 - D / max(R, 1), power=D / max(T, 1),
                              already_over=A0 / max(R, 1)))
    print(pd.DataFrame(crows[-12:]).round(3).to_string(index=False), flush=True)
pd.DataFrame(nrows).to_csv(OUT / "w4_networks.csv", index=False)
pd.DataFrame(crows).to_csv(OUT / "w4_calls.csv", index=False)
pd.concat(keep).to_parquet(ROOT / "data" / "interim" / "w4_scores.parquet", index=False)
