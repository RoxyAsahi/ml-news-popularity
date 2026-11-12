"""S11: how much of popularity is predictable at all from what the dataset records? ("chance" budget)

(a) Twin correlation corrected for non-story twins: r_story = (r_obs - (1 - p) * r_random) / p, with
    p = weighted same-story precision (s9) and r_random = matched-random-neighbour correlation (s3).
    Treating non-story twins like random neighbours makes this an upper value if same-topic pairs
    correlate more than random ones.
(b) Share of the variance of log(1+shares), months >= 2013-07, explained by
    - context: channel x day means (df-adjusted);
    - a past-only HistGB regressor on all features (monthly rolling origin), out-of-sample R^2;
    - both: the regressor's prediction plus channel x day fixed effects (df-adjusted);
    - plus story: the remaining variance times r_story (an upper value for what knowing the story adds).
    The model term is a lower bound on what the features can predict (a better learner could do more) and
    the story term an upper value, so the sum is neither bound: it is the share we can explain. The model R^2
    is out of sample; the channel x day term is in sample, df-adjusted. Week-cluster bootstrap for the model
    and context terms.

Output: results/tables/s11_chance.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from newspop.data import ROOT, feature_set, load
from newspop import stories as st

df = load()
ls = np.log1p(df.shares.to_numpy())
month = df.publish_date.dt.to_period("M").astype(str).to_numpy()
X = df[feature_set(df, "all")].to_numpy(float)
pred = np.full(len(df), np.nan)
for m in sorted(set(month)):
    if m < "2013-07":
        continue
    tr, te = month < m, month == m
    r = HGBR(learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, max_iter=500, random_state=0)
    pred[te] = r.fit(X[tr], ls[tr]).predict(X[te])
ok = ~np.isnan(pred)
y, p = ls[ok], pred[ok]
sst = ((y - y.mean()) ** 2).sum()
r2_model = 1 - ((y - p) ** 2).sum() / sst

chan, day = st.channel(df)[ok], st.day_index(df)[ok]
g = pd.factorize(pd.Series(chan) + "_" + pd.Series(day).astype(str))[0]
k = g.max() + 1


def demean(v):
    s = pd.Series(v)
    return (s - s.groupby(g).transform("mean")).to_numpy()


yd, pd_ = demean(y), demean(p)
r2_ctx_raw = 1 - (yd ** 2).sum() / sst
r2_ctx = 1 - ((yd ** 2).sum() / (len(y) - k)) / (sst / (len(y) - 1))
beta = (yd * pd_).sum() / (pd_ ** 2).sum()
res = yd - beta * pd_
r2_both = 1 - ((res ** 2).sum() / (len(y) - k - 1)) / (sst / (len(y) - 1))

s3 = pd.read_csv(ROOT / "results" / "tables" / "s3_story_similarity.csv").set_index("subset")
s9 = pd.read_csv(ROOT / "results" / "tables" / "s9_precision_power.csv")
prec = s9[s9.check.str.startswith("weighted")].iloc[0]
r_obs, r_lo, r_hi = s3.loc["all twins (sim >= 0.5)", ["r_twin", "r_twin_lo", "r_twin_hi"]]
r_rand = s3.loc["all twins (sim >= 0.5)", "r_matched_random"]
corr = lambda r, pr: (r - (1 - pr) * r_rand) / pr
r_story = corr(r_obs, prec.estimate)
r_story_lo, r_story_hi = corr(r_lo, prec.hi), corr(r_hi, prec.lo)
story_add = (1 - r2_both) * r_story
rows = [
    dict(quantity="twin correlation, observed", value=r_obs, lo=r_lo, hi=r_hi),
    dict(quantity="twin correlation, corrected for non-story twins", value=r_story, lo=r_story_lo, hi=r_story_hi),
    dict(quantity="R2 context (channel x day, adjusted)", value=r2_ctx, lo=np.nan, hi=np.nan),
    dict(quantity="R2 past-only HistGB regressor", value=r2_model, lo=np.nan, hi=np.nan),
    dict(quantity="R2 regressor + channel x day FE (adjusted)", value=r2_both, lo=np.nan, hi=np.nan),
    dict(quantity="+ story (upper value)", value=r2_both + story_add, lo=r2_both + (1 - r2_both) * r_story_lo,
         hi=r2_both + (1 - r2_both) * r_story_hi),
    dict(quantity="not predictable from these data", value=1 - r2_both - story_add, lo=np.nan, hi=np.nan),
]
# week-cluster bootstrap of the model and context terms; the story term is drawn independently
# from a normal approximation to its interval (r_story bounds / 3.92)
wk = st.week_index(df)[ok]
wl, winv = np.unique(wk, return_inverse=True)
idx = [np.where(winv == j)[0] for j in range(len(wl))]
rng = np.random.default_rng(0)
bm, bb, bt = [], [], []
sd_story = (r_story_hi - r_story_lo) / 3.92
for _ in range(300):
    picks = rng.integers(0, len(wl), len(wl))
    sel = np.concatenate([idx[j] for j in picks])
    copy = np.concatenate([np.full(len(idx[j]), c) for c, j in enumerate(picks)])  # a week drawn twice = new groups
    yy, pp, gg = y[sel], p[sel], pd.factorize(pd.Series(g[sel]).astype(str) + "_" + pd.Series(copy).astype(str))[0]
    sst_b = ((yy - yy.mean()) ** 2).sum()
    m_b = 1 - ((yy - pp) ** 2).sum() / sst_b
    dm = lambda v: v - pd.Series(v).groupby(gg).transform("mean").to_numpy()
    yd_b, pd_b = dm(yy), dm(pp)
    be = (yd_b * pd_b).sum() / (pd_b ** 2).sum()
    kk = gg.max() + 1
    both_b = 1 - (((yd_b - be * pd_b) ** 2).sum() / (len(yy) - kk - 1)) / (sst_b / (len(yy) - 1))
    rs = rng.normal(r_story, sd_story)
    bm.append(m_b); bb.append(both_b); bt.append(both_b + (1 - both_b) * rs)
q = lambda v: (np.quantile(v, .025), np.quantile(v, .975))
rows[3]["lo"], rows[3]["hi"] = q(bm)
rows[4]["lo"], rows[4]["hi"] = q(bb)
rows[5]["lo"], rows[5]["hi"] = q(bt)
rows[5]["quantity"] = "+ story (story term is an upper value; model term a lower bound)"
rows[6]["quantity"] = "not explained by this decomposition"
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s11_chance.csv", index=False)
print(R.to_string(index=False))
print("n =", ok.sum(), "raw context R2", round(r2_ctx_raw, 4))
