"""D7: is the weekend premium accounted for by crowding (or exposure) on the day?

Weekend articles get more shares, and weekends are also the days with the fewest articles. The day's volume
counts articles in this dataset, so mechanically it measures exposure (how long an article stays near the top
of the site) as much as competition for readers' attention.

(1) Prediction test: estimate the volume slopes from Monday-Friday articles only (day of week held fixed),
    multiply by how much emptier weekends are, compare with the weekend premium estimated directly.
(2) Overlap: the weekday slope is extrapolated to weekend volumes; how many weekdays are that quiet?
(3) Within-week identification: the same slopes from deviations within the calendar week only (residuals
    demeaned by week; day of week already in W), so slow-news weeks cannot drive them.
(4) Neighbouring days: volume on day t-1 and t+1 next to day t. Crowding / exposure predicts a weaker
    spillover from neighbours; selection of what gets published on the article's own day does not.
(5) Composition: do quiet days carry articles that W predicts to be more popular anyway? Regress the
    out-of-fold prediction of log shares from the article's content (section, topics, keyword history,
    headline vector; no date) on the day's volume, within week, weekday and section.

Outputs: out/d7_weekend_crowding.csv, out/d7_quiet_weekdays.csv, out/d7_crowding_checks.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
dm = lambda a: a - a.mean(0)
wd = (T.weekend == 0).to_numpy()
VOL = ("vol_same", "vol_other")
gap = np.array([T.vol_same[~wd].mean() - T.vol_same[wd].mean(), T.vol_other[~wd].mean() - T.vol_other[wd].mean()])


def within(a, g):
    a = np.asarray(a, float); a2 = a.reshape(len(a), -1)
    return (a2 - pd.DataFrame(a2).groupby(g).transform("mean").to_numpy()).reshape(a.shape)


# ---------------------------------------------------------------- (1) prediction test, (3) within week
rows, chk = [], []
for rep in range(5):
    ry = C.residuals(df, "Y", Y, "title", rep, W)
    R = np.column_stack([C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in VOL])
    b, V = C.cluster_ols(dm(ry[wd]), dm(R[wd]), wk[wd])
    bw_, Vw_ = C.cluster_ols(within(ry[wd], wk[wd]), within(R[wd], wk[wd]), wk[wd])        # weekdays, within week
    ba, Va = C.cluster_ols(within(ry, wk), within(R, wk), wk)                             # all days, within week
    ryb = C.residuals(df, "Y", Y, "title", rep, W, lever="weekend"); rw = C.residuals(df, "weekend", T.weekend.to_numpy(), "title", rep, W, lever="weekend")
    bw, Vw = C.cluster_ols(dm(ryb), dm(rw), wk)
    rows.append(dict(rep=rep, b_same=b[0], b_other=b[1], pred=gap @ b, pred_se=np.sqrt(gap @ V @ gap), direct=bw[0], direct_se=np.sqrt(Vw[0, 0]),
                     pred_within=gap @ bw_, pred_within_se=np.sqrt(gap @ Vw_ @ gap)))
    for spec, bb, VV in [("weekdays only, local trend (as in the prediction test)", b, V), ("weekdays only, within week", bw_, Vw_), ("all days, within week", ba, Va)]:
        for i, l in enumerate(VOL):
            chk.append(dict(check="identification", spec=spec, term=l, rep=rep, theta=bb[i], se=np.sqrt(VV[i, i])))
E = pd.DataFrame(rows)
pc = lambda t: 100 * (np.exp(t) - 1)
pred, pse = C.combine(E.pred, E.pred_se); dirc, dse = C.combine(E.direct, E.direct_se); pw, pwse = C.combine(E.pred_within, E.pred_within_se)
out = pd.DataFrame([
    dict(quantity="articles per section-day, weekend / weekday", value=np.exp(gap[0])),
    dict(quantity="other-section articles per day, weekend / weekday", value=np.exp(gap[1])),
    dict(quantity="volume slope from weekdays only: same section (log-log)", value=E.b_same.median()),
    dict(quantity="volume slope from weekdays only: other sections (log-log)", value=E.b_other.median()),
    dict(quantity="weekend premium predicted from weekday crowding (%)", value=pc(pred), lo=pc(pred - 1.96 * pse), hi=pc(pred + 1.96 * pse)),
    dict(quantity="same, slopes identified within week only (%)", value=pc(pw), lo=pc(pw - 1.96 * pwse), hi=pc(pw + 1.96 * pwse)),
    dict(quantity="weekend premium estimated directly (%)", value=pc(dirc), lo=pc(dirc - 1.96 * dse), hi=pc(dirc + 1.96 * dse))])
out.round(3).to_csv(C.OUT / "d7_weekend_crowding.csv", index=False)
print(out.round(3).to_string(index=False))

# ---------------------------------------------------------------- (2) overlap
D = df.groupby("day").agg(date=("publish_date", "first"), n=("day", "size"), weekend=("is_weekend", "first")).reset_index()
# n counts analysis-sample articles; use the dataset-wide count behind vol_same/vol_other instead
tot = np.exp(df.vol_same) + np.where(np.exp(df.vol_other) <= 1, 0, np.exp(df.vol_other))
D["n_dataset"] = df.assign(tot=tot).groupby("day").tot.first().round().astype(int).to_numpy()
we = D[D.weekend == 1].n_dataset; wdD = D[D.weekend == 0]
q = dict(weekend_days=len(we), weekday_days=len(wdD), weekend_median=we.median(), weekend_p90=we.quantile(0.9), weekend_max=we.max(),
         weekday_median=wdD.n_dataset.median(), weekday_p10=wdD.n_dataset.quantile(0.1), weekday_min=wdD.n_dataset.min(),
         weekdays_at_or_below_weekend_median=int((wdD.n_dataset <= we.median()).sum()), weekdays_at_or_below_weekend_p90=int((wdD.n_dataset <= we.quantile(0.9)).sum()),
         weekdays_at_or_below_weekend_max=int((wdD.n_dataset <= we.max()).sum()))
print(pd.Series(q).round(1).to_string())
Q = wdD[wdD.n_dataset <= we.quantile(0.9)].sort_values("date")[["date", "n_dataset"]]
Q["weekday"] = Q.date.dt.day_name()
Q.to_csv(C.OUT / "d7_quiet_weekdays.csv", index=False)
print(Q.to_string(index=False))

# quiet weekdays (at or below the weekend's 90th percentile; they are US holidays): do they show the premium
# that the slope fitted on all the *other* weekdays predicts? And is the weekday slope the same in the
# quieter and the busier half of weekdays (log-linearity, which the extrapolation to weekends relies on)?
quiet = df.day.isin(wdD[wdD.n_dataset <= we.quantile(0.9)].day).to_numpy()
hol = []
for rep in range(5):
    ry = dm(C.residuals(df, "Y", Y, "title", rep, W)); R = dm(np.column_stack([C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in VOL]))
    fit = wd & ~quiet
    b, V = C.cluster_ols(ry[fit], R[fit], wk[fit])
    hol.append(dict(rep=rep, observed=ry[quiet].mean(), predicted=(R[quiet] @ b).mean(), n=int(quiet.sum()),
                    observed_se=ry[quiet].std() / np.sqrt(len(np.unique(df.day[quiet])))))
    tot_r = R[:, 0] + R[:, 1]
    for lab, m in [("quieter half of weekdays", wd & (tot_r <= np.median(tot_r[wd]))), ("busier half of weekdays", wd & (tot_r > np.median(tot_r[wd])))]:
        bb, VV = C.cluster_ols(dm(ry[m]), dm(R[m]), wk[m])
        for i, l in enumerate(VOL):
            chk.append(dict(check="log-linearity", spec=lab, term=l, rep=rep, theta=bb[i], se=np.sqrt(VV[i, i])))
H = pd.DataFrame(hol)
out = pd.concat([out, pd.DataFrame([
    dict(quantity=f"quiet weekdays ({int(len(np.unique(df.day[quiet])))} holidays, {int(quiet.sum())} articles): observed premium (%)", value=pc(H.observed.median()),
         lo=pc(H.observed.median() - 1.96 * H.observed_se.median()), hi=pc(H.observed.median() + 1.96 * H.observed_se.median())),
    dict(quantity="quiet weekdays: premium predicted from the other weekdays' slope (%)", value=pc(H.predicted.median()))])])
out.round(3).to_csv(C.OUT / "d7_weekend_crowding.csv", index=False)
print(out.tail(2).round(3).to_string(index=False))

# ---------------------------------------------------------------- (4) neighbouring days
NB = ["vol_same_lag1", "vol_same_lead1", "vol_other_lag1", "vol_other_lead1"]
ok = np.isfinite(df[NB].to_numpy()).all(1)
for rep in range(2):
    fo = C.week_folds(df, rep)
    ry = C.residuals(df, "Y", Y, "title", rep, W)
    R = [C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in VOL]
    for nme in NB:
        f = C.CACHE / f"title__week__r{rep}__{nme}sub.npy"
        if f.exists():
            r = np.load(f)
        else:
            pred_ = np.zeros(ok.sum()); Xs, ys, fs = W["title"][ok], df[nme].to_numpy()[ok], fo[ok]
            for k in range(C.N_FOLDS):
                pred_[fs == k] = C.make_learner(seed=k).fit(Xs[fs != k], ys[fs != k]).predict(Xs[fs == k])
            r = ys - pred_; np.save(f, r)
        full = np.full(len(Y), np.nan); full[ok] = r; R.append(full)
    X = np.column_stack(R)[ok]; names = list(VOL) + NB
    b, V = C.cluster_ols(dm(ry[ok]), dm(X), wk[ok])
    for i, l in enumerate(names):
        chk.append(dict(check="neighbouring days", spec="day t, t-1 and t+1 together", term=l, rep=rep, theta=b[i], se=np.sqrt(V[i, i])))
    b, V = C.cluster_ols(dm(ry[ok]), dm(X[:, :2]), wk[ok])
    for i, l in enumerate(VOL):
        chk.append(dict(check="neighbouring days", spec="day t alone, same articles", term=l, rep=rep, theta=b[i], se=np.sqrt(V[i, i])))
    print("neighbours", rep, flush=True)

# ---------------------------------------------------------------- (5) composition
for rep in range(2):
    fo = C.week_folds(df, rep)
    g = C.oof("Y", Y, W["content"], fo, f"content__week__r{rep}")          # W-predicted log shares, content only
    Z = np.column_stack([pd.get_dummies(df.week, dtype=float).to_numpy()[:, 1:], pd.get_dummies(df.dow, dtype=float).to_numpy()[:, 1:],
                         pd.get_dummies(df.chan, dtype=float).to_numpy()[:, 1:]])
    res = lambda v: v - LinearRegression().fit(Z, v).predict(Z)
    X = np.column_stack([res(T[l].to_numpy()) for l in VOL])
    for lab, yy in [("predicted log shares from content (composition)", g), ("actual log shares", Y)]:
        b, V = C.cluster_ols(res(yy), X, wk)
        for i, l in enumerate(VOL):
            chk.append(dict(check="composition", spec=lab + ", week + weekday + section FE", term=l, rep=rep, theta=b[i], se=np.sqrt(V[i, i])))
    chk.append(dict(check="composition", spec="out-of-fold R2 of the content-only predictor", term="", rep=rep, theta=1 - np.mean((Y - g) ** 2) / Y.var(), se=np.nan))
K = pd.DataFrame(chk); rows = []
for key, gdf in K.groupby(["check", "spec", "term"], sort=False):
    if gdf.se.isna().all():
        rows.append(dict(check=key[0], spec=key[1], term=key[2], theta=gdf.theta.mean())); continue
    th, se = C.combine(gdf.theta.to_numpy(), gdf.se.to_numpy())
    rows.append(dict(check=key[0], spec=key[1], term=key[2], theta=th, se=se, z=th / se, pct_per_doubling=C.pct(th, "log"),
                     pct_lo=C.pct(th - 1.96 * se, "log"), pct_hi=C.pct(th + 1.96 * se, "log")))
KK = pd.DataFrame(rows); KK.round(4).to_csv(C.OUT / "d7_crowding_checks.csv", index=False)
print(KK.round(3).to_string(index=False))
