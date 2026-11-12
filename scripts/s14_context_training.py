"""S14: does training on within-context contrasts improve within-story ranking on Mashable?

On Upworthy (u2), demeaning the training target within the publication week moved observational models
closer to the randomized within-test ranking. The Mashable analogue: the decision is which of the day's
articles (or which twin of a story) to promote, but standard models are trained to separate popular from
unpopular articles across all contexts. Past-only monthly rolling origin as in s2; variants:
  clf       HGB classifier on shares >= 1,400 (s2's hgb_all, from the score cache);
  reg       HGB regressor on log shares;
  reg-cd    HGB regressor on log shares demeaned within channel x day (training months only);
  reg-cw    the same within channel x week;
  twin-pair HGB regressor on feature differences of earlier twin pairs (both orientations), scoring only
            twin pairs (antisymmetrised: f(a - b) - f(b - a)).
Metrics: pooled AUC, AUC within channel x day, twin concordance on the binary label and on pairs whose
shares differ by > 2x (story-cluster bootstrap).
Output: results/tables/s14_context_training.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from newspop import anatomy as an, stories as st
from newspop.data import ROOT, feature_set, load

df = load()
y = (df.shares >= 1400).astype(int).to_numpy()
ls = np.log1p(df.shares.to_numpy())
X = df[feature_set(df, "all")].to_numpy(dtype=float)
week, day, chan = st.week_index(df), st.day_index(df), st.channel(df)
cd = pd.factorize(pd.Series(chan) + "_" + pd.Series(day).astype(str))[0]
cw = pd.factorize(pd.Series(chan) + "_" + pd.Series(week).astype(str))[0]
month = df.publish_date.dt.to_period("M").astype(str).to_numpy()
S = pd.read_csv(ROOT / "data" / "processed" / "scores.csv")
P = st.clean_twins(df)
sid = st.story_ids(len(df), P)
a_all, b_all = P.i.to_numpy(), P.j.to_numpy()
pmonth = np.maximum(month[a_all], month[b_all])   # a pair is usable for training once both are published
PARAMS = dict(learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, max_iter=500, random_state=0)


def demeaned(g, tr):
    t = pd.Series(ls[tr]).groupby(g[tr]).transform("mean").to_numpy()
    return ls[tr] - t


scores = {"clf": S.hgb_all.to_numpy(), "reg": np.full(len(df), np.nan), "reg-cd": np.full(len(df), np.nan),
          "reg-cw": np.full(len(df), np.nan)}
pair_score = np.full(len(P), np.nan)
for m in sorted(set(month)):
    if m < "2013-07":
        continue
    tr, te = month < m, month == m
    for name, target in [("reg", ls[tr]), ("reg-cd", demeaned(cd, tr)), ("reg-cw", demeaned(cw, tr))]:
        scores[name][te] = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], target).predict(X[te])
    ptr, pte = pmonth < m, pmonth == m
    if ptr.sum() > 50 and pte.any():
        D = X[a_all[ptr]] - X[b_all[ptr]]
        t = ls[a_all[ptr]] - ls[b_all[ptr]]
        mdl = HistGradientBoostingRegressor(**PARAMS).fit(np.r_[D, -D], np.r_[t, -t])
        Dt = X[a_all[pte]] - X[b_all[pte]]
        pair_score[pte] = mdl.predict(Dt) - mdl.predict(-Dt)
    print(m, flush=True)

rows = []
for name, s in scores.items():
    m = ~np.isnan(s)
    pooled = an.pooled_auc_boot(y[m], s[m], np.zeros(m.sum(), int), week[m])
    wcd = an.within_auc_boot(y[m], s[m], cd[m], week[m])
    ok = m[a_all] & m[b_all] & ~np.isnan(pair_score)   # same pairs as the twin-pair model, for comparability
    tw = an.pair_concordance(s, y, a_all[ok], b_all[ok], sid[a_all[ok]])
    tw2 = an.pair_concordance(s, ls, a_all[ok], b_all[ok], sid[a_all[ok]], min_gap=np.log(2))
    rows.append(dict(model=name, pooled_auc=pooled[0], pooled_lo=pooled[1], pooled_hi=pooled[2],
                     chan_day_auc=wcd[0], chan_day_lo=wcd[1], chan_day_hi=wcd[2],
                     twin_auc=tw[0], twin_lo=tw[1], twin_hi=tw[2], twin_pairs=tw[3],
                     twin_conc2x=tw2[0], twin_conc2x_lo=tw2[1], twin_conc2x_hi=tw2[2]))
    print(rows[-1], flush=True)
# twin-pair model: concordance of the pair score with the outcome difference
ok = ~np.isnan(pair_score)
a, b, ps = a_all[ok], b_all[ok], pair_score[ok]
# pair_concordance expects item scores; build a pseudo-score so that s[a] - s[b] = ps
pseudo = np.full(len(df), np.nan)
res = []
for target, gap in [(y, 0.0), (ls, np.log(2))]:
    dt = target[a] - target[b]
    use = np.abs(dt) > gap if gap > 0 else dt != 0
    cl = sid[a][use]
    agree = np.where(ps[use] == 0, 0.5, (np.sign(ps[use]) == np.sign(dt[use])).astype(float))
    labs, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(0)
    sums, cnts = np.bincount(inv, agree), np.bincount(inv)
    boots = [sums[k].sum() / cnts[k].sum() for k in (rng.integers(0, len(labs), len(labs)) for _ in range(1000))]
    res.append((agree.mean(), *np.quantile(boots, [.025, .975]), use.sum()))
rows.append(dict(model="twin-pair", twin_auc=res[0][0], twin_lo=res[0][1], twin_hi=res[0][2], twin_pairs=res[0][3],
                 twin_conc2x=res[1][0], twin_conc2x_lo=res[1][1], twin_conc2x_hi=res[1][2]))
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s14_context_training.csv", index=False)
print(R.to_string(index=False))
