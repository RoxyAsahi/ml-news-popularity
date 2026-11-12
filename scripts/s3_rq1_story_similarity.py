"""S3 (RQ1): do same-story articles share their fate? Twin correlation vs matched random neighbours,
split by publication gap and channel; first-vs-later difference (cannibalisation check);
nested variance shares (week, channel x week) with omega-squared style df adjustment.

Residual log shares = log(1+shares) after channel x week and weekday fixed effects (so neither the
publication week, the section nor the weekday can create similarity). first_minus_later uses pairs
with a positive day gap only. Baseline for each twin pair (i, j): j is replaced by
a random article from the same channel and the same day as j (20 draws, averaged).
CIs: story-cluster bootstrap (stories = connected components of the twin graph).

Outputs: results/tables/s3_story_similarity.csv, results/tables/s3_variance_shares.csv
"""
import numpy as np
import pandas as pd

from newspop.data import ROOT, load
from newspop import stories as st

df = load()
ls = np.log1p(df.shares.to_numpy())
week, day, chan = st.week_index(df), st.day_index(df), st.channel(df)
cw = pd.Series(chan) + "_" + pd.Series(week).astype(str)
import statsmodels.formula.api as smf
res = smf.ols("ls ~ C(cw) + C(wd)", pd.DataFrame({"ls": ls, "cw": cw, "wd": df.publish_date.dt.dayofweek})).fit().resid.to_numpy()

P = st.clean_twins(df)
P["story"] = st.story_ids(len(df), P)[P.i]
P["later"] = np.where(P["first"] == P.i, P.j, P.i)
rng = np.random.default_rng(0)

# matched random neighbours: same channel, same day as the second article
key = pd.Series(chan) + "_" + pd.Series(day).astype(str)
members = key.groupby(key).indices


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def baseline_draw(Q, rng):
    j2 = np.array([rng.choice(members[key[j]]) for j in Q.j])
    ok = j2 != Q.i.to_numpy()
    return Q.i.to_numpy()[ok], j2[ok]


def summarise(name, Q, n_boot=500):
    a, b = Q.i.to_numpy(), Q.j.to_numpy()
    r_twin = corr(res[a], res[b])
    base = [corr(res[x], res[y]) for x, y in (baseline_draw(Q, rng) for _ in range(20))]
    r_base = float(np.mean(base))
    gap = np.where(Q.dt.to_numpy() > 0, res[Q["first"].to_numpy()] - res[Q.later.to_numpy()], np.nan)
    stories = Q.story.to_numpy()
    labels, inv = np.unique(stories, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(labels))]
    bt, bd, bg = [], [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.integers(0, len(labels), len(labels))])
        rt = corr(res[a[sel]], res[b[sel]])
        bt.append(rt)
        bd.append(rt - r_base)
        bg.append(np.nanmean(gap[sel]))
    q = lambda v: np.quantile(v, [0.025, 0.975])
    return dict(subset=name, pairs=len(Q), stories=len(labels), r_twin=r_twin, r_twin_lo=q(bt)[0],
                r_twin_hi=q(bt)[1], r_matched_random=r_base, diff=r_twin - r_base, diff_lo=q(bd)[0],
                diff_hi=q(bd)[1], first_minus_later=np.nanmean(gap), fml_lo=q(bg)[0], fml_hi=q(bg)[1])


rows = [summarise("all twins (sim >= 0.5)", P)]
for lo, hi, lab in [(0, 0, "same day"), (1, 2, "1-2 days apart"), (3, 7, "3-7 days apart")]:
    rows.append(summarise(lab, P[(P.dt >= lo) & (P.dt <= hi)]))
rows.append(summarise("same channel", P[P.same_channel]))
rows.append(summarise("different channel", P[~P.same_channel]))
for th in [0.6, 0.7]:
    rows.append(summarise(f"sim >= {th}", P[P.sim >= th]))
# control: random same-week pairs ordered by date. Does the earlier article also win there?
# (rules out a within-week time trend; CI from a week-cluster bootstrap)
wk = pd.Series(week).groupby(week).indices
a = rng.integers(0, len(df), 200_000)
b = np.array([rng.choice(wk[week[x]]) for x in a])
ok = day[a] != day[b]
first = np.where(day[a] < day[b], a, b)[ok]
later = np.where(day[a] < day[b], b, a)[ok]
dlt = res[first] - res[later]
wlab, winv = np.unique(week[first], return_inverse=True)
wsum, wcnt = np.bincount(winv, weights=dlt), np.bincount(winv)
wb = []
for _ in range(500):
    pick = rng.integers(0, len(wlab), len(wlab))
    wb.append(wsum[pick].sum() / wcnt[pick].sum())
rows.append(dict(subset="control: random same-week pairs", pairs=int(ok.sum()), first_minus_later=dlt.mean(),
                 fml_lo=np.quantile(wb, 0.025), fml_hi=np.quantile(wb, 0.975)))

R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s3_story_similarity.csv", index=False)
print(R.to_string(index=False))


# nested variance shares (adjusted for the number of group means fitted)
def explained(groups, y):
    g = pd.Series(y).groupby(groups)
    fitted = g.transform("mean").to_numpy()
    k = g.ngroups
    ss_between = ((fitted - y.mean()) ** 2).sum()
    ss_within = ((y - fitted) ** 2).sum()
    ms_within = ss_within / (len(y) - k)
    return (ss_between - (k - 1) * ms_within) / ((y - y.mean()) ** 2).sum()


V = pd.DataFrame([
    {"level": "publication week", "share_of_variance": explained(week, ls)},
    {"level": "channel x week (incl. week)", "share_of_variance": explained(cw.to_numpy(), ls)},
    {"level": "channel x day (incl. above)", "share_of_variance": explained(key.to_numpy(), ls)},
]).round(4)
V.to_csv(ROOT / "results" / "tables" / "s3_variance_shares.csv", index=False)
print(V.to_string(index=False))
