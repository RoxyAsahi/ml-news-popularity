"""U1: does holding the story fixed recover randomized headline effects? A ladder of designs on the
Upworthy Research Archive (Matias et al. 2021), where the truth is known from randomized A/B tests.

Outcome: logit click-through rate of a headline package. Features: 15 headline features (Pattern-lexicon
sentiment as in the Mashable data, length, punctuation, pronouns), standardized.
Designs (each estimates the effect of every feature on logit CTR):
  T  truth: randomized within-test comparisons (fixed effects for test x image x lede, so only the
     headline varies), test-clustered bootstrap;
  L0 naive cross-section: one random package per test, OLS across tests;
  L1 + week fixed effects;
  L1e + week FE + 50 SVD dimensions of the test's headline text (topic/embedding control, as tried by
     Banerjee & Urminsky);
  L2 story twins (our Mashable design): pairs of different tests (different articles) whose headline
     text has TF-IDF cosine >= 0.5 and that started <= 7 days apart; regression of the within-pair
     difference in logit CTR on feature differences (one random package per test);
  L3 same-article retests: pairs of tests of the identical article (same lede) run at different times.
Agreement with the truth over the 15 features: Pearson r, RMSE, sign agreement where the truth is
significant; excess squared bias (squared deviation minus both sampling variances, averaged over features)
and coverage of the truth by each design's own 95% intervals (a valid but noisy design covers ~95%); per feature z = (b - t) / sqrt(var_b + var_t), the number of
features with |z| > 1.96 and a chi-square test (15 df) of no bias. Point estimates average 20 random package draws; intervals bootstrap clusters (tests, or
twin-graph components), averaging 20 fresh draws per replicate, and resample the truth independently.

Usage: python scripts/u1_design_ladder.py [exploratory|confirmatory]   (analysis fixed on exploratory)
Output: results/tables/u1_ladder_<split>.csv, results/tables/u1_coefs_<split>.csv
"""
import sys

import numpy as np
from scipy.stats import chi2
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from newspop.data import ROOT
from newspop import upworthy as up

split = sys.argv[1] if len(sys.argv) > 1 else "exploratory"
B, DRAWS = 200, 20
rng = np.random.default_rng(0)
d = up.load(split)
F = up.headline_features(d.headline)
F = (F - F.mean()) / F.std()
names = list(F.columns)
X = F.to_numpy()
y = d.logit_ctr.to_numpy()
k = len(names)


def ols(Xm, yv):
    return np.linalg.lstsq(Xm, yv, rcond=None)[0]


def demean(A, g):
    A = np.asarray(A, float)
    s = np.zeros((g.max() + 1,) + A.shape[1:])
    np.add.at(s, g, A)
    c = np.bincount(g).reshape((-1,) + (1,) * (A.ndim - 1))
    return A - (s / c)[g]


# ---- truth: within test x image x lede
cell = pd.factorize(d.test.astype(str) + "|" + d.eyecatcher_id.astype(str) + "|" + d.lede_clean)[0]
nh = pd.Series(d.headline.to_numpy()).groupby(cell).transform("nunique").to_numpy()
w = nh >= 2
cw = pd.factorize(cell[w])[0]
Xt, yt, tt = demean(X[w], cw), demean(y[w], cw), d.test.to_numpy()[w]
truth = ols(Xt, yt)
tl, tinv = np.unique(tt, return_inverse=True)
tidx = [np.where(tinv == j)[0] for j in range(len(tl))]
truth_b = np.array([ols(Xt[s], yt[s]) for s in
                    (np.concatenate([tidx[j] for j in rng.integers(0, len(tl), len(tl))]) for _ in range(B))])
print(f"truth: {w.sum()} packages in {len(tl)} tests", flush=True)

# ---- test level
tests = d.groupby("test").agg(t=("t", "min"), week=("test_week", "first"), lede=("lede_clean", "first"),
                              text=("headline", lambda s: " ".join(s.unique()))).reset_index()
tpos = {t: i for i, t in enumerate(tests.test)}
pk = [np.where(d.test.to_numpy() == t)[0] for t in tests.test]
V = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(tests.text)
E = TruncatedSVD(50, random_state=0).fit_transform(V)
WK = pd.get_dummies(tests.week.astype(str), drop_first=True).to_numpy(float)
S = (V @ V.T).tocoo()
day = (tests.t - tests.t.min()).dt.days.to_numpy()
m = S.row < S.col
i, j, s = S.row[m], S.col[m], S.data[m]
same_lede = (tests.lede.to_numpy()[i] == tests.lede.to_numpy()[j]) & (tests.lede.to_numpy()[i] != "")
twin = (s >= 0.5) & ~same_lede & (np.abs(day[i] - day[j]) <= 7)
retest = same_lede & (day[i] != day[j])
pairs = {}
for name, mask in [("L2 story twins", twin), ("L3 same-article retests", retest)]:
    a, b = i[mask], j[mask]
    first = np.where(day[a] <= day[b], a, b)
    second = np.where(day[a] <= day[b], b, a)
    comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(tests),) * 2), directed=False)[1]
    pairs[name] = (first, second, comp[first])
    print(name, len(a), "pairs", flush=True)


def draw(r):
    return np.array([p[r.integers(0, len(p))] for p in pk])


def fit_design(name, pick, sel_tests=None, sel_pairs=None):
    if name.startswith("L2") or name.startswith("L3"):
        f, s_, _ = pairs[name]
        if sel_pairs is not None:
            f, s_ = f[sel_pairs], s_[sel_pairs]
        D = X[pick[f]] - X[pick[s_]]
        return ols(np.c_[D, np.ones(len(D))], y[pick[f]] - y[pick[s_]])[:k]
    idx = np.arange(len(tests)) if sel_tests is None else sel_tests
    Xm = X[pick[idx]]
    if name == "L0 naive cross-section":
        M = np.c_[Xm, np.ones(len(idx))]
    elif name == "L1 + week FE":
        M = np.c_[Xm, np.ones(len(idx)), WK[idx]]
    else:
        M = np.c_[Xm, np.ones(len(idx)), WK[idx], E[idx]]
    return ols(M, y[pick[idx]])[:k]


def agree(b, t, tb):
    sig = (np.quantile(tb, .025, 0) > 0) | (np.quantile(tb, .975, 0) < 0)
    return dict(r=np.corrcoef(b, t)[0, 1], rmse=np.sqrt(np.mean((b - t) ** 2)),
                sign_agree_sig=np.mean(np.sign(b[sig]) == np.sign(t[sig])) if sig.any() else np.nan)


designs = ["L0 naive cross-section", "L1 + week FE", "L1e + week FE + text embedding", "L2 story twins",
           "L3 same-article retests"]
rows, coefs = [], [dict(design="T truth (randomized)", **dict(zip(names, truth)))]
for name in designs:
    est = np.mean([fit_design(name, draw(rng)) for _ in range(DRAWS)], 0)
    coefs.append(dict(design=name, **dict(zip(names, est))))
    boots, bcoef = [], []
    for bi in range(B):
        # each replicate resamples clusters once and averages DRAWS package draws, like the point estimate
        if name.startswith("L2") or name.startswith("L3"):
            comp = pairs[name][2]
            labs, inv = np.unique(comp, return_inverse=True)
            members = [np.where(inv == c)[0] for c in range(len(labs))]
            sel = np.concatenate([members[c] for c in rng.integers(0, len(labs), len(labs))])
            bb = np.mean([fit_design(name, draw(rng), sel_pairs=sel) for _ in range(DRAWS)], 0)
        else:
            sel = rng.integers(0, len(tests), len(tests))
            bb = np.mean([fit_design(name, draw(rng), sel_tests=sel) for _ in range(DRAWS)], 0)
        boots.append(agree(bb, truth_b[bi], truth_b))
        bcoef.append(bb)
    bt = pd.DataFrame(boots)
    bcoef = np.array(bcoef)
    a = agree(est, truth, truth_b)
    # bias beyond noise: mean over features of (b - t)^2 - var(b) - var(t); coverage of the truth by the
    # design's own 95% interval; how much of the design's coefficients' spread is noise
    var_b, var_t = bcoef.var(0), truth_b.var(0)
    a["bias2_excess"] = float(np.mean((est - truth) ** 2 - var_b - var_t))
    lo_, hi_ = np.quantile(bcoef, .025, 0), np.quantile(bcoef, .975, 0)
    a["coverage_of_truth"] = float(np.mean((truth >= lo_) & (truth <= hi_)))
    a["mean_se"] = float(np.sqrt(var_b).mean())
    z = (est - truth) / np.sqrt(var_b + var_t)
    a["n_features_biased"] = int((np.abs(z) > 1.96).sum())  # ~0.75 of 15 expected without bias
    a["chi2_bias_p"] = float(1 - chi2.cdf((z ** 2).sum(), k))
    for f in range(k):
        coefs.append(dict(design=name + " [lo]", **dict(zip(names, lo_)))) if f == 0 else None
        coefs.append(dict(design=name + " [hi]", **dict(zip(names, hi_)))) if f == 0 else None
    n = len(pairs[name][0]) if name in pairs else len(tests)
    rows.append(dict(design=name, n=n, **a, **{f"{c}_lo": bt[c].quantile(.025) for c in bt},
                     **{f"{c}_hi": bt[c].quantile(.975) for c in bt}))
    print(name, {c: round(v, 3) for c, v in a.items()}, flush=True)
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / f"u1_ladder_{split}.csv", index=False)
C = pd.DataFrame(coefs).set_index("design").T.round(4)
C["truth_lo"], C["truth_hi"] = np.quantile(truth_b, .025, 0).round(4), np.quantile(truth_b, .975, 0).round(4)
C.to_csv(ROOT / "results" / "tables" / f"u1_coefs_{split}.csv")
print(R.to_string(index=False))
print(C.to_string())
