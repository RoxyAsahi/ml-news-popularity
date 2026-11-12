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
    The rest is not predictable from these data ("chance" relative to the information available).

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
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s11_chance.csv", index=False)
print(R.to_string(index=False))
print("n =", ok.sum(), "raw context R2", round(r2_ctx_raw, 4))
