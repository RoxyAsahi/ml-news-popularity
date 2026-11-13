"""Direction A, controls asked for by the internal review (teacher view), on the current archive panel.

 0  URL repair, external check: early count vs final count under the repaired mapping and under the file as distributed.
 1  How much of the headline number is bookkeeping? Per horizon: share of articles already past 1,400 at the look,
    AUC among those not yet past it, and prediction of the REMAINING growth log1p(final) - log1p(count now).
 2  Szabo-Huberman baseline: log final = a + log count (slope fixed at 1) and the 3-term linear version, past-only,
    against the HGB models.
 3  Same quantity? Homepage total vs sum of the six networks; late homepage total vs the dataset's final count.
 4  Actual age of the last observation per nominal horizon.
 6  Facebook rule with the threshold chosen on earlier months only (their 90th percentile at 2-6 h).
 7  Conformal selection: (a) false-discovery proportion month by month, (b) power counted over ALL blockbusters of
    the month including never-sighted articles, (c) calibration on the most recent 1 or 2 months only (drift test).
Scores for 1 and 7 come from w4 (data/interim/w4_scores.parquet). Output: results/tables/w6_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COUNT_COLS, HGB_PARAMS, NETS, ROOT, STATE_COLS, VEL_COLS, early_state, features, load_fixed,  # noqa: E402
                    load_obs)

OUT = ROOT / "results" / "tables"
FIRST = "2013-04"
pd.set_option("display.width", 220)
df = load_fixed()
n = len(df)
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float)
y = np.log1p(s)
month = df.date.dt.to_period("M").astype(str).to_numpy()
O = load_obs(df, lists_only=True)
O["final"] = s[O.row.to_numpy()]
last_month = O.ts.max().strftime("%Y-%m")

# ------------------------------------------------------------------ 0: URL repair, external check
raw = pd.read_csv(ROOT / "data" / "raw" / "OnlineNewsPopularity.csv"); raw.columns = raw.columns.str.strip()
raw["art_id"] = raw.url.str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2}/[^/]+)")[0].str.lower()
raw["date"] = pd.to_datetime("2015-01-08") - pd.to_timedelta(raw.timedelta, unit="D")
Oall = pd.read_parquet(ROOT / "data" / "interim" / "wb_obs.parquet")
Oall = Oall[(Oall.age_h > 0) & Oall.total.notna()]
maps = {"repaired": df.dropna(subset=["art_id"]).drop_duplicates("art_id").set_index("art_id")[["shares", "date"]],
        "as_distributed": raw.dropna(subset=["art_id"]).drop_duplicates("art_id").set_index("art_id")[["shares", "date"]]}
r0 = []
for nm, mp in maps.items():
    j = Oall.join(mp, on="art_id", how="inner")
    for per, mk in [("before 2013-06-06", j.date < "2013-06-06"), ("from 2013-06-06", j.date >= "2013-06-06")]:
        o = j[mk]
        e = o[(o.age_h >= 6) & (o.age_h <= 48)].groupby("art_id").tail(1)
        r0.append(dict(mapping=nm, period=per, sightings=len(o), early_above_final=float((o.total > o.shares * 1.05 + 100).mean()),
                       articles=len(e), corr_log_early_final=float(np.corrcoef(np.log1p(e.total), np.log1p(e.shares))[0, 1]),
                       spearman=float(spearmanr(e.total, e.shares)[0])))
R0 = pd.DataFrame(r0); R0.to_csv(OUT / "w6_repair_check.csv", index=False)
print("0. URL repair, external check (early = last sighting at 6-48 h)\n", R0.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 3: same quantity?
gap = (O.total - O[NETS].sum(axis=1))
print("\n3. homepage total minus sum of six networks: share exactly 0 = %.3f, median %.0f, 95th pct %.0f" %
      ((gap == 0).mean(), gap.median(), gap.quantile(.95)))
late = O[O.age_h >= 24].groupby("row").tail(1).copy()
late["ratio"] = late.total / late.final.clip(lower=1)
late["band"] = pd.cut(late.age_h, [24, 36, 48, 72, 96, 1e4], right=False)
late["half"] = np.where(late.ts < "2013-07-01", "2013H1", np.where(late.ts < "2014-01-01", "2013H2", "2014"))
R3 = late.groupby("band", observed=True).ratio.describe(percentiles=[.1, .25, .5, .75, .9]).drop(columns=["mean", "std", "min", "max"])
R3["above_final"] = late.groupby("band", observed=True).ratio.apply(lambda v: (v > 1.05).mean())
print("late homepage total / dataset final count, by age of the sighting (hours):\n", R3.round(3).to_string())
print(late.groupby("half").ratio.median().round(3).to_dict(), flush=True)
R3.reset_index().assign(band=lambda d: d.band.astype(str)).to_csv(OUT / "w6_same_quantity.csv", index=False)

# ------------------------------------------------------------------ 1, 2, 4: bookkeeping, baselines, age
S = pd.read_parquet(ROOT / "data" / "interim" / "w4_scores.parquet")


def reg():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


rows = []
for h in sorted(S.h.unique()):
    E = early_state(O, h)
    Sh = S[S.h == h].set_index("row")
    idx = E.index.to_numpy(); mi = month[idx]; yi = y[idx]; si = s[idx]
    lc = E.l_total.to_numpy(); la = np.log(E.age.to_numpy().clip(min=.05))
    g = yi - lc  # remaining growth on the log scale
    A_e = E[COUNT_COLS + STATE_COLS + VEL_COLS].to_numpy(dtype=float); A_e[:, np.isnan(A_e).all(0)] = 0.0
    mats = {"prepub": X[idx], "early": A_e, "both": np.c_[X[idx], A_e]}
    gs = {k: np.full(len(idx), np.nan) for k in mats}
    sh1 = np.full(len(idx), np.nan); sh3 = np.full(len(idx), np.nan)
    L = np.c_[lc, la, lc * la]
    for m in sorted(set(mi)):
        tr, te = mi < m, mi == m
        if m < FIRST or tr.sum() < 300 or te.sum() == 0:
            continue
        sh1[te] = lc[te] + (yi[tr] - lc[tr]).mean()
        sh3[te] = LinearRegression().fit(L[tr], yi[tr]).predict(L[te])
        for k, A in mats.items():
            gs[k][te] = reg().fit(A[tr], g[tr]).predict(A[te])
    ok = np.isin(idx, Sh.index.to_numpy()) & ~np.isnan(sh1)
    sc = {"szabo_huberman": sh1[ok], "linear3": sh3[ok], **{k: Sh[k].reindex(idx[ok]).to_numpy() for k in ["prepub", "early", "both"]}}
    cur = np.expm1(lc[ok]); below = cur < 1400; lab = si[ok] >= 1400
    age = E.age.to_numpy()[ok]
    base = dict(h=h, n=int(ok.sum()), already_1400=float((~below).mean()), p1400=float(lab.mean()),
                p1400_not_yet=float(lab[below].mean()), age_q25=np.quantile(age, .25), age_median=np.median(age),
                age_q75=np.quantile(age, .75))
    for k, z in sc.items():
        r = dict(**base, scorer=k, auc1400=roc_auc_score(lab, z), auc_not_yet=roc_auc_score(lab[below], z[below]),
                 r2_final=r2_score(yi[ok], z), spearman_final=spearmanr(z, yi[ok])[0])
        if k in gs:
            r["r2_growth"] = r2_score(g[ok], gs[k][ok]); r["spearman_growth"] = spearmanr(gs[k][ok], g[ok])[0]
        elif k == "szabo_huberman":
            r["r2_growth"] = r2_score(g[ok], np.full(ok.sum(), 0.0) + (sh1[ok] - lc[ok])); r["spearman_growth"] = np.nan
        else:
            r["r2_growth"] = r2_score(g[ok], sh3[ok] - lc[ok]); r["spearman_growth"] = spearmanr(sh3[ok] - lc[ok], g[ok])[0]
        rows.append(r)
    print(pd.DataFrame(rows[-5:]).drop(columns=["age_q25", "age_q75", "p1400"]).round(3).to_string(index=False), flush=True)
R1 = pd.DataFrame(rows); R1.to_csv(OUT / "w6_bookkeeping_baselines.csv", index=False)
print("\n4. actual age (h) of the last observation per nominal horizon")
print(R1.drop_duplicates("h")[["h", "n", "age_q25", "age_median", "age_q75", "already_1400"]].round(2).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 6: Facebook rule, past-only threshold
w = O[(O.age_h >= 2) & (O.age_h < 6)].groupby("row").tail(1)
wm = month[w.row.to_numpy()]; tp = sel = pos = tot = 0; thr_used = []
for m in sorted(set(wm)):
    if m < FIRST:
        continue
    c = np.quantile(s[month < m], .95); thr = np.quantile(w.facebook[wm < m], .90); te = wm == m
    pick = te & (w.facebook.to_numpy() >= thr)
    tp += (w.final[pick] > c).sum(); sel += pick.sum(); pos += (w.final[te] > c).sum(); tot += te.sum(); thr_used.append(thr)
print("\n6. Facebook rule, threshold = 90th percentile of the Facebook count at 2-6 h in earlier months "
      "(%.0f..%.0f): picks %.1f%% of articles, precision %.3f (base %.3f), recall %.3f" %
      (min(thr_used), max(thr_used), 100 * sel / tot, tp / sel, pos / tot, tp / pos), flush=True)
pd.DataFrame([dict(thr_min=min(thr_used), thr_max=max(thr_used), picked=sel / tot, precision=tp / sel, base=pos / tot,
                   recall=tp / pos)]).to_csv(OUT / "w6_facebook_rule.csv", index=False)

# ------------------------------------------------------------------ 7: conformal selection, deeper
crows, mrows = [], []
for h in (6.0, 12.0, 24.0):
    Sh = S[S.h == h]; idx = Sh.row.to_numpy(); mi = month[idx]; si = s[idx]; ms = sorted(set(mi))
    for k in ("early", "both", "prepub"):
        v = Sh[k].to_numpy()
        for win in (0, 2, 1):  # 0 = all earlier months
            for q in (0.1, 0.2, 0.3):
                R = D = T = Tall = 0; fd = []
                for j, m in enumerate(ms[2:], start=2):
                    calm = ms[:j] if win == 0 else ms[max(0, j - win):j]
                    cal = np.isin(mi, calm); te = mi == m; c = np.quantile(s[month < m], .95)
                    vn = np.sort(v[cal & (si <= c)])
                    p = (1 + (len(vn) - np.searchsorted(vn, v[te], side="left"))) / (cal.sum() + 1)
                    o = np.argsort(p); r_ = np.arange(1, te.sum() + 1)
                    good = np.flatnonzero(p[o] <= q * r_ / te.sum())
                    r = good.max() + 1 if len(good) else 0
                    hit = (si[te][o[:r]] > c).sum()
                    R += r; D += hit; T += (si[te] > c).sum(); Tall += (s[month == m] > c).sum()
                    fd.append(1 - hit / r if r else 0.0)  # FDP is 0 by convention when nothing is selected
                    if win == 0 and k == "early":
                        mrows.append(dict(h=h, q=q, month=m, n_test=int(te.sum()), shortlisted=int(r), fdp=fd[-1]))
                crows.append(dict(h=h, scorer=k, cal_months="all" if win == 0 else win, q=q, shortlisted=R,
                                  fdp_pooled=(1 - D / R) if R else np.nan, fdr_mean_over_months=float(np.mean(fd)),
                                  fdp_month_max=float(np.max(fd)), power_sighted=D / T, power_all_articles=D / Tall))
C = pd.DataFrame(crows); C.to_csv(OUT / "w6_conformal.csv", index=False)
M = pd.DataFrame(mrows); M.to_csv(OUT / "w6_conformal_by_month.csv", index=False)
print("\n7. conformal selection of top-5% articles")
print(C.round(3).to_string(index=False))
print("\nmonth by month, early signals, all earlier months as calibration, h = 12:")
print(M[M.h == 12].pivot(index="month", columns="q", values=["shortlisted", "fdp"]).round(2).to_string())
