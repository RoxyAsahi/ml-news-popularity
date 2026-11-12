"""S12: does the headline text (URL slug) add ranking skill, and where? Past-only throughout.

slug_score: for each month m >= 2013-02, a ridge regression of log(1+shares) on TF-IDF of slug words
(unigrams + bigrams, vectorizer fitted on articles before m only) trained on all articles before m,
applied to the articles of m. HistGB is then refitted monthly (m >= 2013-07) on articles from
2013-02 to m-1, with and without slug_score as an extra feature (same rows, same settings).
Compares pairwise AUC overall, within channel x day and within same-story twins (same channel and all),
with joint week- (or story-) cluster bootstraps of the difference.

Output: results/tables/s12_slug_model.csv, data/processed/slug_scores.csv
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from newspop.data import ROOT, feature_set, load
from newspop import anatomy as an, stories as st

df = load()
y = (df.shares >= 1400).astype(int).to_numpy()
ls = np.log1p(df.shares.to_numpy())
month = df.publish_date.dt.to_period("M").astype(str).to_numpy()
text = st.slug_text(df.url).fillna("").to_numpy()
X = df[feature_set(df, "all")].to_numpy(float)

slug = np.full(len(df), np.nan)
for m in sorted(set(month)):
    if m < "2013-02":
        continue
    tr, te = month < m, month == m
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    V = vec.fit_transform(text[tr])
    r = Ridge(alpha=10.0).fit(V, ls[tr] - ls[tr].mean())
    slug[te] = r.predict(vec.transform(text[te]))
print("slug score corr with log shares (out of sample):", round(np.corrcoef(slug[~np.isnan(slug)], ls[~np.isnan(slug)])[0, 1], 3), flush=True)

base, plus = np.full(len(df), np.nan), np.full(len(df), np.nan)
Xp = np.c_[X, slug]
from newspop.models import fit, get_model
for m in sorted(set(month)):
    if m < "2013-07":
        continue
    tr, te = (month < m) & (month >= "2013-02"), month == m
    for out, M in [(base, X), (plus, Xp)]:
        clf = fit(get_model("hgb", seed=0, **an.HGB_PARAMS), M[tr], y[tr])
        out[te] = clf.predict_proba(M[te])[:, 1]
    print(m, flush=True)
pd.DataFrame({"url": df.url, "slug_score": slug, "hgb_base": base, "hgb_slug": plus}).to_csv(
    ROOT / "data" / "processed" / "slug_scores.csv", index=False)

week, day, chan = st.week_index(df), st.day_index(df), st.channel(df)
ok = ~np.isnan(base)
rng = np.random.default_rng(0)
rows = []


def ctx(name, groups):
    g1 = an.group_u(y[ok], base[ok], groups[ok])
    g2 = an.group_u(y[ok], plus[ok], groups[ok])
    cl = pd.Series(week[ok], index=groups[ok]).groupby(level=0).first()
    p1 = g1.groupby(cl.reindex(g1.index).to_numpy()).sum()
    p2 = g2.groupby(cl.reindex(g2.index).to_numpy()).sum().reindex(p1.index)
    a1, a2 = p1.u.sum() / p1.pairs.sum(), p2.u.sum() / p2.pairs.sum()
    b = []
    for _ in range(1000):
        k = rng.integers(0, len(p1), len(p1))
        b.append(p2.u.to_numpy()[k].sum() / p2.pairs.to_numpy()[k].sum() - p1.u.to_numpy()[k].sum() / p1.pairs.to_numpy()[k].sum())
    rows.append(dict(context=name, auc_base=a1, auc_slug=a2, diff=a2 - a1, lo=np.quantile(b, .025), hi=np.quantile(b, .975)))


ctx("all pairs (same week)", week)  # week-nested so the bootstrap is valid; pooled differs by < 0.002
ctx("same channel x day", pd.factorize(pd.Series(chan) + "_" + pd.Series(day).astype(str))[0])
P = st.clean_twins(df)
P["story"] = st.story_ids(len(df), P)[P.i]
for name, Q in [("same-story twins", P), ("same-story twins, same channel", P[P.same_channel])]:
    Q = Q[ok[Q.i] & ok[Q.j]]
    a, b_ = Q.i.to_numpy(), Q.j.to_numpy()
    keep = y[a] != y[b_]
    pos, neg = np.where(y[a] == 1, a, b_)[keep], np.where(y[a] == 1, b_, a)[keep]
    v1 = np.where(base[pos] == base[neg], .5, (base[pos] > base[neg]).astype(float))
    v2 = np.where(plus[pos] == plus[neg], .5, (plus[pos] > plus[neg]).astype(float))
    lab, inv = np.unique(Q.story.to_numpy()[keep], return_inverse=True)
    s1, s2, c = np.bincount(inv, v1), np.bincount(inv, v2), np.bincount(inv)
    bt = [(lambda k: (s2[k].sum() - s1[k].sum()) / c[k].sum())(rng.integers(0, len(lab), len(lab))) for _ in range(1000)]
    rows.append(dict(context=name, auc_base=v1.mean(), auc_slug=v2.mean(), diff=v2.mean() - v1.mean(),
                     lo=np.quantile(bt, .025), hi=np.quantile(bt, .975)))
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s12_slug_model.csv", index=False)
print(R.to_string(index=False))
