"""S4 (RQ2): which content effects survive topic control? Twin-difference vs cross-section.

Content features (standardised on all articles): log body length, log title length, log images,
log videos, log links, log internal links, average word length, number of keywords, body
subjectivity, share of positive words, share of negative words (global_rate_*), title subjectivity,
title polarity. Negative tone is measured by global_rate_negative_words (share of all words that
are negative), NOT by polarity, to avoid the polarity / negative-rate collinearity.

Cross-section: OLS of log(1+shares) on the features + channel x week + weekday fixed effects
(within transformation); week-cluster bootstrap.
Twins: OLS of the within-pair difference (earlier minus later) on feature differences + a constant
(the earlier-article advantage) + channel-difference dummies; story-cluster bootstrap.
Difference (twin - cross-section): independent bootstraps combined, 95% interval; BH-adjusted
p-values for the twin coefficients. Variants: sim >= 0.6, same channel only, similar length
(|delta log length| < 0.5).

Outputs: results/tables/s4_twin_effects.csv
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from newspop.data import ROOT, load
from newspop import stories as st

df = load()
ls = np.log1p(df.shares.to_numpy())
week, chan = st.week_index(df), st.channel(df)
F = pd.DataFrame({
    "log body length": np.log1p(df.n_tokens_content), "log title length": np.log1p(df.n_tokens_title),
    "log images": np.log1p(df.num_imgs), "log videos": np.log1p(df.num_videos),
    "log links": np.log1p(df.num_hrefs), "log internal links": np.log1p(df.num_self_hrefs),
    "average word length": df.average_token_length, "number of keywords": df.num_keywords,
    "body subjectivity": df.global_subjectivity, "share positive words": df.global_rate_positive_words,
    "share negative words": df.global_rate_negative_words, "title subjectivity": df.title_subjectivity,
    "title polarity": df.title_sentiment_polarity})
F = (F - F.mean()) / F.std()
names = list(F.columns)
X = F.to_numpy()
rng = np.random.default_rng(0)
B = 1000

# ---- cross-section, within channel x week, weekday dummies
cw = pd.factorize(pd.Series(chan) + "_" + pd.Series(week).astype(str))[0]
WD = pd.get_dummies(df.publish_date.dt.dayofweek, drop_first=True).to_numpy(float)
Z = np.c_[X, WD]


def demean(A, g):
    A = np.asarray(A, float)
    s = np.zeros((g.max() + 1,) + A.shape[1:])
    np.add.at(s, g, A)
    c = np.bincount(g)
    return A - (s / c.reshape((-1,) + (1,) * (A.ndim - 1)))[g]


def cs_fit(idx):
    g = pd.factorize(cw[idx])[0]
    Zd, yd = demean(Z[idx], g), demean(ls[idx], g)
    return np.linalg.lstsq(Zd, yd, rcond=None)[0][:len(names)]


cs = cs_fit(np.arange(len(df)))
wk_members = pd.Series(np.arange(len(df))).groupby(week).apply(np.array).to_list()
cs_boot = np.array([cs_fit(np.concatenate([wk_members[k] for k in rng.integers(0, len(wk_members), len(wk_members))]))
                    for _ in range(B)])


# ---- twins
def twin_design(Q):
    later = np.where(Q["first"] == Q.i, Q.j, Q.i)
    first = Q["first"].to_numpy()
    D = X[first] - X[later]
    C = pd.get_dummies(pd.Series(chan[first]), dtype=float).reindex(columns=sorted(set(chan)), fill_value=0).to_numpy() \
        - pd.get_dummies(pd.Series(chan[later]), dtype=float).reindex(columns=sorted(set(chan)), fill_value=0).to_numpy()
    C = C[:, 1:]
    W = np.c_[D, np.ones(len(Q)), C]
    return W, ls[first] - ls[later], Q.story.to_numpy()


def tw_fit(W, dy):
    return np.linalg.lstsq(W, dy, rcond=None)[0]


P = st.clean_twins(df)
P["story"] = st.story_ids(len(df), P)[P.i]
variants = {"main (sim >= 0.5)": P, "sim >= 0.6": P[P.sim >= 0.6], "same channel": P[P.same_channel],
            "similar length": P[np.abs(np.log1p(df.n_tokens_content.to_numpy()[P.i]) - np.log1p(df.n_tokens_content.to_numpy()[P.j])) < 0.5]}
rows = []
for vname, Q in variants.items():
    W, dy, story = twin_design(Q)
    est = tw_fit(W, dy)
    labels, inv = np.unique(story, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(labels))]
    boot = np.array([tw_fit(W[s], dy[s]) for s in
                     (np.concatenate([members[k] for k in rng.integers(0, len(labels), len(labels))]) for _ in range(B))])
    k = len(names)
    p_tw = np.minimum(1, 2 * np.minimum((boot[:, :k] > 0).mean(0), (boot[:, :k] < 0).mean(0)))
    diff_boot = boot[:, :k] - cs_boot
    p_diff = np.minimum(1, 2 * np.minimum((diff_boot > 0).mean(0), (diff_boot < 0).mean(0)))
    q_tw = multipletests(np.maximum(p_tw, 1 / B), method="fdr_bh")[1]
    for f in range(k):
        rows.append(dict(variant=vname, feature=names[f], pairs=len(Q), cross_section=cs[f],
                         cs_lo=np.quantile(cs_boot[:, f], .025), cs_hi=np.quantile(cs_boot[:, f], .975),
                         twin=est[f], tw_lo=np.quantile(boot[:, f], .025), tw_hi=np.quantile(boot[:, f], .975),
                         twin_q_bh=q_tw[f], diff=est[f] - cs[f], diff_lo=np.quantile(diff_boot[:, f], .025),
                         diff_hi=np.quantile(diff_boot[:, f], .975), diff_p=p_diff[f]))
    rows.append(dict(variant=vname, feature="earlier-article advantage", pairs=len(Q), twin=est[k],
                     tw_lo=np.quantile(boot[:, k], .025), tw_hi=np.quantile(boot[:, k], .975)))
    print(vname, "done", flush=True)
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s4_twin_effects.csv", index=False)
pd.set_option("display.width", 250)
print(R[R.variant == "main (sim >= 0.5)"].drop(columns=["variant", "pairs"]).to_string(index=False))
for v in list(variants)[1:]:
    print(v, R[(R.variant == v) & R.feature.isin(["log body length", "body subjectivity", "share negative words", "log images", "log links"])][["feature", "twin", "tw_lo", "tw_hi"]].to_string(index=False))
# twin feature similarity (format heterogeneity)
for f in ["log body length", "body subjectivity", "log images"]:
    print("within-twin correlation of", f, round(np.corrcoef(F[f].to_numpy()[P.i], F[f].to_numpy()[P.j])[0, 1], 3))
