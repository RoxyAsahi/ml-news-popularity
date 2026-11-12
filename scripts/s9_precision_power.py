"""S9: two checks asked for in review.

(a) Twin precision weighted by the similarity mix of the twins actually used. The 180 annotated
    pairs are stratified (30 per band), but most clean twins (sim >= 0.5) fall in the lowest band,
    so the pooled 78% overstates precision. Weighted precision = sum_b w_b * p_b with w_b = share
    of clean twins in band b and p_b = series-filtered precision in band b; normal-approximation CI
    from the stratified variance sum_b w_b^2 p_b (1 - p_b) / n_b.
    Caveat: the series filter was designed while inspecting these same 180 pairs.
(b) Does fixing the story cost skill beyond channel x day? Difference between the pairwise AUC of
    past-only HistGB on same-story same-channel twins (story-cluster bootstrap) and within
    channel x day (week-cluster bootstrap), with independent bootstraps combined (as an
    equivalence-style bound).

Output: results/tables/s9_precision_power.csv
"""
import numpy as np
import pandas as pd

from newspop.data import ROOT, load
from newspop import anatomy as an, stories as st

df = load()
rows = []

# (a) weighted precision
prec = pd.read_csv(ROOT / "results" / "tables" / "s1b_twin_precision.csv")
prec = prec[prec.series_filtered & prec.subset.str.startswith("band")]
P = st.clean_twins(df)
edges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
w, p, n = [], [], []
for lo, hi in edges:
    lab = f"band [{lo},{min(hi, 1.0):.1f})"
    r = prec[prec.subset == lab].iloc[0]
    w.append(((P.sim >= lo) & (P.sim < hi)).mean())
    p.append(r.precision)
    n.append(r.n)
w, p, n = map(np.array, (w, p, n))
est = float((w * p).sum())
se = float(np.sqrt((w ** 2 * p * (1 - p) / n).sum()))
for (lo, hi), wi, pi in zip(edges, w, p):
    rows.append(dict(check="share of clean twins in band", item=f"[{lo},{min(hi, 1.0):.1f})", estimate=wi, lo=np.nan,
                     hi=np.nan, note=f"band precision {pi:.3f}"))
rows.append(dict(check="weighted twin precision (same story)", item="sim >= 0.5", estimate=est,
                 lo=est - 1.96 * se, hi=est + 1.96 * se, note="weights = similarity mix of clean twins"))

# (b) twin vs channel x day
y = (df.shares >= 1400).astype(int).to_numpy()
S = pd.read_csv(ROOT / "data" / "processed" / "scores.csv")
s = S["hgb_all"].to_numpy()
m = ~np.isnan(s)
week, day, chan = st.week_index(df), st.day_index(df), st.channel(df)
g = pd.factorize(pd.Series(chan) + "_" + pd.Series(day).astype(str))[0]
G = an.group_u(y[m], s[m], g[m])
cl = pd.Series(week[m], index=g[m]).groupby(level=0).first().reindex(G.index)
per = G.groupby(cl.to_numpy()).sum()
U, Pn = per.u.to_numpy(), per.pairs.to_numpy()
rng = np.random.default_rng(0)
B = 2000
cd = U.sum() / Pn.sum()
cd_b = np.array([(lambda k: U[k].sum() / Pn[k].sum())(rng.integers(0, len(U), len(U))) for _ in range(B)])

P["story"] = st.story_ids(len(df), P)[P.i]
Q = P[P.same_channel & m[P.i] & m[P.j]]
a, b = Q.i.to_numpy(), Q.j.to_numpy()
keep = y[a] != y[b]
pos = np.where(y[a] == 1, a, b)[keep]
neg = np.where(y[a] == 1, b, a)[keep]
v = np.where(s[pos] == s[neg], 0.5, (s[pos] > s[neg]).astype(float))
lab, inv = np.unique(Q.story.to_numpy()[keep], return_inverse=True)
sums, cnts = np.bincount(inv, weights=v), np.bincount(inv)
tw = v.mean()
tw_b = np.array([(lambda k: sums[k].sum() / cnts[k].sum())(rng.integers(0, len(lab), len(lab))) for _ in range(B)])
d = tw_b - cd_b
rows.append(dict(check="AUC twins (same channel) minus channel x day", item="hgb_all past-only", estimate=tw - cd,
                 lo=np.quantile(d, 0.025), hi=np.quantile(d, 0.975),
                 note=f"twins {tw:.3f} ({keep.sum()} pairs), channel x day {cd:.3f}"))
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s9_precision_power.csv", index=False)
print(R.to_string(index=False))
