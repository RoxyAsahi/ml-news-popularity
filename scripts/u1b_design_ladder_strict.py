"""U1b: the design ladder of u1, revised after review (u1 is kept as the version fixed on the exploratory split).

Changes against u1:
  * Pairs are defined by the article's lede text, not by exact lede equality. In the archive, most pairs of
    tests with similar headlines are the same article with a slightly edited lede (lede TF-IDF cosine >= 0.94
    for 90% of them), so u1's "story twins" were mostly same-article pairs.
      L2 story twins (strict): headline TF-IDF cosine >= 0.5, started <= 7 days apart, both ledes non-empty
         and lede cosine < 0.3 (different articles about the same story);
      L3 same-article retests: lede cosine >= 0.8, both ledes non-empty, started on different days.
  * Out-of-split truth: the randomized truth is estimated on the other two splits of the archive (random
    partitions of tests by the archive's authors), so truth and design share no tests and no noise.
  * A cross-section at the sample size of the strict twins (L0n), to show what the naive design looks like
    at the same noise level.
  * Excess squared bias with an approximate bootstrap interval, and per-feature equivalence (TOST): a
    feature counts as recovered when the 90% bootstrap interval of (design - truth) lies within +-MARGIN.

Usage: python scripts/u1b_design_ladder_strict.py [exploratory|confirmatory|holdout]
Output: results/tables/u1b_ladder_<split>.csv, results/tables/u1b_coefs_<split>.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from newspop import upworthy as up
from newspop.data import ROOT

SPLITS = ["exploratory", "confirmatory", "holdout"]
split = sys.argv[1] if len(sys.argv) > 1 else "confirmatory"
B, DRAWS, MARGIN = 200, 20, 0.005
rng = np.random.default_rng(0)

# features standardized on the pooled archive so that truth and designs share one scale
parts = {s: up.load(s) for s in SPLITS}
Fall = {s: up.headline_features(parts[s].headline) for s in SPLITS}
pool = pd.concat(Fall.values())
mu, sd = pool.mean(), pool.std()
names = list(pool.columns)
k = len(names)


def ols(Xm, yv):
    return np.linalg.lstsq(Xm, yv, rcond=None)[0]


def demean(A, g):
    A = np.asarray(A, float)
    s = np.zeros((g.max() + 1,) + A.shape[1:])
    np.add.at(s, g, A)
    c = np.bincount(g).reshape((-1,) + (1,) * (A.ndim - 1))
    return A - (s / c)[g]


# ---- truth from the other two splits: within test x image x lede
other = [s for s in SPLITS if s != split]
dt = pd.concat([parts[s] for s in other], ignore_index=True)
Xo = ((pd.concat([Fall[s] for s in other], ignore_index=True) - mu) / sd).to_numpy()
cell = pd.factorize(dt.test.astype(str) + "|" + dt.eyecatcher_id.astype(str) + "|" + dt.lede_clean)[0]
nh = pd.Series(dt.headline.to_numpy()).groupby(cell).transform("nunique").to_numpy()
w = nh >= 2
cw = pd.factorize(cell[w])[0]
Xt, yt, tt = demean(Xo[w], cw), demean(dt.logit_ctr.to_numpy()[w], cw), dt.test.to_numpy()[w]
truth = ols(Xt, yt)
tl, tinv = np.unique(tt, return_inverse=True)
tidx = [np.where(tinv == j)[0] for j in range(len(tl))]
truth_b = np.array([ols(Xt[s], yt[s]) for s in
                    (np.concatenate([tidx[j] for j in rng.integers(0, len(tl), len(tl))]) for _ in range(B))])
print(f"truth ({'+'.join(other)}): {w.sum()} packages in {len(tl)} tests", flush=True)

# ---- designs on this split
d = parts[split]
X = ((Fall[split] - mu) / sd).to_numpy()
y = d.logit_ctr.to_numpy()
tests = d.groupby("test").agg(t=("t", "min"), week=("test_week", "first"), lede=("lede_clean", "first"),
                              text=("headline", lambda s: " ".join(s.unique()))).reset_index()
pk = [np.where(d.test.to_numpy() == t)[0] for t in tests.test]
V = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(tests.text)
L = TfidfVectorizer(sublinear_tf=True).fit_transform(tests.lede)
E = TruncatedSVD(50, random_state=0).fit_transform(V)
WK = pd.get_dummies(tests.week.astype(str), drop_first=True).to_numpy(float)
day = (tests.t - tests.t.min()).dt.days.to_numpy()
has_lede = tests.lede.to_numpy() != ""

S = (V @ V.T).tocoo()
m = (S.row < S.col) & (S.data >= 0.5)
i, j = S.row[m], S.col[m]
lc = np.asarray(L[i].multiply(L[j]).sum(1)).ravel()
twin = (np.abs(day[i] - day[j]) <= 7) & has_lede[i] & has_lede[j] & (lc < 0.3)
print(f"headline-similar pairs <= 7 days with ledes: {((np.abs(day[i]-day[j]) <= 7) & has_lede[i] & has_lede[j]).sum()}, "
      f"of which lede cosine >= 0.8: {((np.abs(day[i]-day[j]) <= 7) & has_lede[i] & has_lede[j] & (lc >= .8)).sum()}",
      flush=True)
ri, rj = [], []
for a0 in range(0, len(tests), 3000):
    M = (L[a0:a0 + 3000] @ L.T).tocoo()
    r, c = M.row + a0, M.col
    keep = (r < c) & (M.data >= 0.8) & has_lede[r] & has_lede[c] & (day[r] != day[c])
    ri.append(r[keep]), rj.append(c[keep])
ri, rj = np.concatenate(ri), np.concatenate(rj)

pairs = {}
for name, (a, b) in [("L2 story twins (strict)", (i[twin], j[twin])), ("L3 same-article retests", (ri, rj))]:
    first = np.where(day[a] <= day[b], a, b)
    second = np.where(day[a] <= day[b], b, a)
    comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(tests),) * 2), directed=False)[1]
    pairs[name] = (first, second, comp[first])
    print(name, len(a), "pairs in", len(np.unique(comp[first])), "clusters", flush=True)
n_small = len(pairs["L2 story twins (strict)"][0])


def draw(r):
    return np.array([p[r.integers(0, len(p))] for p in pk])


def fit_design(name, pick, sel=None):
    if name in pairs:
        f, s_, _ = pairs[name]
        if sel is not None:
            f, s_ = f[sel], s_[sel]
        D = X[pick[f]] - X[pick[s_]]
        return ols(np.c_[D, np.ones(len(D))], y[pick[f]] - y[pick[s_]])[:k]
    idx = np.arange(len(tests)) if sel is None else sel
    Xm = X[pick[idx]]
    if name.startswith("L0"):
        M = np.c_[Xm, np.ones(len(idx))]
    elif name == "L1 + week FE":
        M = np.c_[Xm, np.ones(len(idx)), WK[idx]]
    else:
        M = np.c_[Xm, np.ones(len(idx)), WK[idx], E[idx]]
    return ols(M, y[pick[idx]])[:k]


def resample(name, r):
    if name in pairs:
        comp = pairs[name][2]
        labs, inv = np.unique(comp, return_inverse=True)
        members = [np.where(inv == c)[0] for c in range(len(labs))]
        return np.concatenate([members[c] for c in r.integers(0, len(labs), len(labs))])
    if name.startswith("L0n"):
        return r.choice(len(tests), n_small, replace=False)  # a fresh small sample of tests each time
    return r.integers(0, len(tests), len(tests))


designs = ["L0 naive cross-section", f"L0n cross-section, n = {n_small}", "L1 + week FE",
           "L1e + week FE + text embedding", "L2 story twins (strict)", "L3 same-article retests"]
var_t = truth_b.var(0)
rows, coefs = [], [dict(design="T truth (randomized, other splits)", **dict(zip(names, truth)))]
for name in designs:
    if name.startswith("L0n"):
        est = np.mean([fit_design(name, draw(rng), resample(name, rng)) for _ in range(DRAWS)], 0)
    else:
        est = np.mean([fit_design(name, draw(rng)) for _ in range(DRAWS)], 0)
    bcoef = np.array([np.mean([fit_design(name, draw(rng), sel) for _ in range(DRAWS)], 0)
                      for sel in (resample(name, rng) for _ in range(B))])
    var_b = bcoef.var(0)
    diff_b = bcoef - truth_b                      # independent resamples of design and truth
    lo_, hi_ = np.quantile(bcoef, .025, 0), np.quantile(bcoef, .975, 0)
    z = (est - truth) / np.sqrt(var_b + var_t)
    dev2_b = (diff_b ** 2).mean(1)               # E = bias^2 + 2(var_b + var_t) on average
    noise = (var_b + var_t).mean()
    tost = (np.quantile(diff_b, .05, 0) > -MARGIN) & (np.quantile(diff_b, .95, 0) < MARGIN)
    a = dict(design=name, n=len(pairs[name][0]) if name in pairs else (n_small if name.startswith("L0n") else len(tests)),
             r=np.corrcoef(est, truth)[0, 1], rmse=np.sqrt(np.mean((est - truth) ** 2)),
             rmse_lo=np.sqrt(np.quantile((diff_b ** 2).mean(1), .025)), rmse_hi=np.sqrt(np.quantile(dev2_b, .975)),
             mean_se=np.sqrt(var_b).mean(),
             bias2_excess=np.mean((est - truth) ** 2) - noise,
             bias2_lo=np.quantile(dev2_b, .025) - 2 * noise, bias2_hi=np.quantile(dev2_b, .975) - 2 * noise,
             coverage_of_truth=np.mean((truth >= lo_) & (truth <= hi_)),
             n_features_biased=int((np.abs(z) > 1.96).sum()), chi2_bias_p=1 - chi2.cdf((z ** 2).sum(), k),
             n_features_equivalent=int(tost.sum()))
    rows.append(a)
    coefs += [dict(design=name, **dict(zip(names, est))), dict(design=name + " [lo]", **dict(zip(names, lo_))),
              dict(design=name + " [hi]", **dict(zip(names, hi_)))]
    print({c: (round(v, 4) if isinstance(v, float) else v) for c, v in a.items()}, flush=True)

R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / f"u1b_ladder_{split}.csv", index=False)
C = pd.DataFrame(coefs).set_index("design").T.round(4)
C["truth_lo"], C["truth_hi"] = np.quantile(truth_b, .025, 0).round(4), np.quantile(truth_b, .975, 0).round(4)
C.to_csv(ROOT / "results" / "tables" / f"u1b_coefs_{split}.csv")
print(R.to_string(index=False))
