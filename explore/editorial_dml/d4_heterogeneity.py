"""D4: do the lever effects differ between sections, years, or kinds of article?

(a) Section and year: the main-specification residuals, with the lever's residual interacted with the
    section (or the year); Wald test that all sections share one effect (week-clustered).
(b) Causal forest (Athey, Tibshirani & Wager 2019; econml.grf) on the same residuals: tau(x) over section,
    LDA topics, date, keyword history and the other levers. Honest check of the heterogeneity it finds:
    forests are trained on half of the weeks and scored on the other half (and vice versa); then
      - calibration regression  Yres ~ a * mean(tau) * Tres + b * (tau_hat - mean(tau)) * Tres
        (b = 1: heterogeneity is real and correctly scaled; b = 0: the forest found noise);
      - effect within thirds of predicted tau_hat (sorted group effects).

Outputs: out/d4_by_section.csv, out/d4_by_year.csv, out/d4_by_period.csv, out/d4_forest_calibration.csv, out/d4_forest_groups.csv,
         out/d4_forest_importance.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
A = [l for l in C.LEVERS if l != "weekend"]
REPS = 5
dm = lambda a: a - a.mean(0)
SEC = sorted(set(df.chan))
YEAR = df.publish_date.dt.year.to_numpy()
# three periods; 2013-06-07 is where the raw file's url column starts to be shifted (irrelevant after realignment,
# kept as a stability check asked for by the main research thread)
PERIOD = np.where(df.publish_date < "2013-06-07", "a: before 2013-06-07", np.where(YEAR == 2013, "b: rest of 2013", "c: 2014"))


def res(lv, rep):
    ry = C.residuals(df, "Y", Y, "title", rep, W, lever=lv)
    rt = C.residuals(df, lv, T[lv].to_numpy(), "title", rep, W, lever=lv)
    return dm(ry), dm(rt)


def by_group(lv, labels, levels):
    B, SE, P = [], [], []
    for rep in range(REPS):
        ry, rt = res(lv, rep)
        X = np.column_stack([rt * (labels == s) for s in levels])
        b, V = C.cluster_ols(ry, X, wk)
        k = len(levels)
        Rm = np.column_stack([np.ones(k - 1), -np.eye(k - 1)])
        q = (Rm @ b) @ np.linalg.solve(Rm @ V @ Rm.T, Rm @ b)
        B.append(b); SE.append(np.sqrt(np.diag(V))); P.append(1 - stats.chi2.cdf(q, k - 1))
    th, se = C.combine(B, SE)
    return th, se, float(np.median(P))


rows_s, rows_y, rows_p = [], [], []
for lv, (label, kind) in C.LEVERS.items():
    th, se, p = by_group(lv, df.chan.to_numpy(), SEC)
    for s, a, b in zip(SEC, th, se):
        rows_s.append(dict(lever=lv, section=s, n=int((df.chan == s).sum()), theta=a, se=b, pct=C.pct(a, kind),
                           pct_lo=C.pct(a - 1.96 * b, kind), pct_hi=C.pct(a + 1.96 * b, kind), p_equal=p))
    th, se, p = by_group(lv, YEAR, [2013, 2014])
    for s, a, b in zip([2013, 2014], th, se):
        rows_y.append(dict(lever=lv, year=s, theta=a, se=b, pct=C.pct(a, kind), pct_lo=C.pct(a - 1.96 * b, kind),
                           pct_hi=C.pct(a + 1.96 * b, kind), p_equal=p))
    PL = sorted(set(PERIOD))
    th, se, p = by_group(lv, PERIOD, PL)
    for s, a, b in zip(PL, th, se):
        rows_p.append(dict(lever=lv, period=s, n=int((PERIOD == s).sum()), theta=a, se=b, pct=C.pct(a, kind),
                           pct_lo=C.pct(a - 1.96 * b, kind), pct_hi=C.pct(a + 1.96 * b, kind), p_equal=p))
    BA = np.where(df.publish_date < "2013-06-07", "before", "after")
    th, se, p = by_group(lv, BA, ["before", "after"])
    rows_p.append(dict(lever=lv, period="before vs after 2013-06-07 (test only)", n=len(BA), theta=th[0] - th[1], se=np.nan,
                       pct=np.nan, pct_lo=np.nan, pct_hi=np.nan, p_equal=p))
Rs, Ry = pd.DataFrame(rows_s), pd.DataFrame(rows_y)
Rp = pd.DataFrame(rows_p); Rp.round(4).to_csv(C.OUT / "d4_by_period.csv", index=False)
print(Rp[Rp.se.notna()].pivot(index="lever", columns="period", values="pct").reindex(list(C.LEVERS)).round(1).to_string())
print(Rp[Rp.se.isna()].set_index("lever").p_equal.round(4).to_string())
Rs.round(4).to_csv(C.OUT / "d4_by_section.csv", index=False); Ry.round(4).to_csv(C.OUT / "d4_by_year.csv", index=False)
print(Rs.pivot(index="lever", columns="section", values="pct").reindex(list(C.LEVERS)).round(1).to_string())
print(Rs.groupby("lever").p_equal.first().reindex(list(C.LEVERS)).round(4).to_string())
print(Ry.pivot(index="lever", columns="year", values="pct").reindex(list(C.LEVERS)).round(1).to_string())
print(Ry.groupby("lever").p_equal.first().reindex(list(C.LEVERS)).round(4).to_string())
if len(sys.argv) > 1 and sys.argv[1] == "nofores":
    raise SystemExit

# ---------------------------------------------------------------- causal forest
from econml.grf import CausalForest

g = C.groups(df)
Xn = [f"sec_{s}" for s in SEC] + g["lda"] + ["day", "kw_avg_avg", "kw_max_avg", "kw_min_avg", "num_keywords"] + \
     ["imgs", "videos", "length", "title_len", "links_out", "links_self", "weekend"]
Xall = np.column_stack([np.column_stack([(df.chan == s).to_numpy(float) for s in SEC]), df[g["lda"]].to_numpy(),
                        df.day, np.log1p(df.kw_avg_avg), np.log1p(df.kw_max_avg), np.log1p(df.kw_min_avg.clip(lower=0)), df.num_keywords,
                        T[["imgs", "videos", "length", "title_len", "links_out", "links_self", "weekend"]].to_numpy()])
cal, grp, imp = [], [], []
half = (pd.Series(wk).map(dict(zip(np.unique(wk), np.random.default_rng(7).permutation(len(np.unique(wk))) % 2)))).to_numpy()
for lv, (label, kind) in C.LEVERS.items():
    f = C.CACHE / f"forest_tau__{lv}.npy"
    ry, rt = res(lv, 0)
    keep = [i for i, nme in enumerate(Xn) if nme != lv]  # the lever itself is not a moderator
    X = Xall[:, keep]
    if f.exists():
        tau = np.load(f); fi = np.load(str(f).replace("tau", "imp"))
    else:
        tau, fi = np.zeros(len(ry)), np.zeros(len(keep))
        for h in (0, 1):
            cf = CausalForest(n_estimators=1000, min_samples_leaf=40, max_samples=0.45, max_features=0.5, honest=True,
                              inference=False, n_jobs=C.THREADS, random_state=h)
            cf.fit(X[half != h], rt[half != h], ry[half != h])
            tau[half == h] = cf.predict(X[half == h]).ravel(); fi += cf.feature_importances_ / 2
        np.save(f, tau); np.save(str(f).replace("tau", "imp"), fi)
    tbar = np.array([tau[half == h].mean() for h in (0, 1)])[half]
    b, V = C.cluster_ols(ry, np.column_stack([tbar * rt, (tau - tbar) * rt]), wk)
    cal.append(dict(lever=lv, mean_forecast=b[0], mean_se=np.sqrt(V[0, 0]), differential_forecast=b[1], diff_se=np.sqrt(V[1, 1]),
                    diff_p_one_sided=1 - stats.norm.cdf(b[1] / np.sqrt(V[1, 1])), sd_tau_hat=tau.std()))
    third = np.zeros(len(tau), int)
    for h in (0, 1):
        m = half == h; third[m] = pd.qcut(tau[m], 3, labels=False)
    b3, V3 = C.cluster_ols(ry, np.column_stack([rt * (third == q) for q in range(3)]), wk)
    for q in range(3):
        s = np.sqrt(V3[q, q])
        grp.append(dict(lever=lv, third=["lowest predicted", "middle", "highest predicted"][q], mean_tau_hat=tau[third == q].mean(),
                        theta=b3[q], se=s, pct=C.pct(b3[q], kind), pct_lo=C.pct(b3[q] - 1.96 * s, kind), pct_hi=C.pct(b3[q] + 1.96 * s, kind)))
    d = b3[2] - b3[0]; sd = np.sqrt(V3[2, 2] + V3[0, 0] - 2 * V3[0, 2])
    grp.append(dict(lever=lv, third="highest minus lowest", theta=d, se=sd, pct=np.nan, pct_lo=np.nan, pct_hi=np.nan))
    top = np.argsort(-fi)[:5]
    imp.append(dict(lever=lv, **{f"top{i + 1}": f"{np.array(Xn)[keep][j]} ({fi[j]:.2f})" for i, j in enumerate(top)}))
    print(lv, "calibration", np.round(b, 2), np.round(np.sqrt(np.diag(V)), 2), flush=True)
pd.DataFrame(cal).round(4).to_csv(C.OUT / "d4_forest_calibration.csv", index=False)
pd.DataFrame(grp).round(4).to_csv(C.OUT / "d4_forest_groups.csv", index=False)
pd.DataFrame(imp).to_csv(C.OUT / "d4_forest_importance.csv", index=False)
print(pd.DataFrame(cal).round(3).to_string(index=False)); print(pd.DataFrame(grp).round(3).to_string(index=False))
print(pd.DataFrame(imp).to_string(index=False))
