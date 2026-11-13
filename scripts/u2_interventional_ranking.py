"""U2: do popularity models trained on observational data rank headlines the way randomized experiments do?

Setting. A publisher trains a click model on its archive: each article ran with one headline, and the model
learns which headlines go with high click-through. The decision the model is used for is different: for one
article, which of several candidate headlines will draw more clicks. On the Upworthy Research Archive both
can be measured: across tests (observational) and within a test, where headlines were randomized
(interventional).

Training signals (none uses the randomized within-test variation, except the oracle):
  OBS     one random package per test, pointwise regression of logit CTR on the headline (standard practice);
  OBS-FE  the same with the target demeaned within publication week (a standard confounder adjustment);
  NAT     natural pairs: two tests of the same article (lede TF-IDF cosine >= 0.8) run on different days, one
          random package each; regression of the CTR difference on the headline difference (a within-article
          contrast that occurs in observational archives when content is re-promoted);
  ORACLE  randomized within-test contrasts (packages sharing test, image and lede), demeaned - upper bound.
Representations: 15 interpretable features (H15), TF-IDF word uni+bigrams, mean GloVe-100 vectors + H15.
Models: ridge (alpha by internal CV); for OBS also gradient boosting on H15+GloVe (a typical popularity model).

Evaluation on held-out articles (5-fold CV grouped by article = connected components of same-article tests):
  across-test Spearman correlation of prediction with logit CTR (observational skill);
  within-test pairwise accuracy: share of pairs of headlines in the same cell ordered correctly, over all pairs
  and over pairs whose CTRs differ significantly (|z| > 1.96, two-proportion test);
  decision value: CTR of the headline the model picks in each cell, relative to the cell mean (in %), compared
  with the observed best (an optimistic ceiling) - how many extra clicks the model would buy.

Usage: python scripts/u2_interventional_ranking.py [exploratory|confirmatory|holdout]
Output: results/tables/u2_ranking_<split>.csv
"""
import gzip
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from newspop import upworthy as up
from newspop.data import ROOT

split = sys.argv[1] if len(sys.argv) > 1 else "exploratory"
rng = np.random.default_rng(0)
ALPHAS = np.logspace(-1, 4, 11)
# TF-IDF runs out of memory (16 GB) on the confirmatory split; it was the weakest representation on the
# exploratory split, so the larger splits use the two dense representations
REPS = ["H15", "TF-IDF", "GloVe+H15"] if split == "exploratory" else ["H15", "GloVe+H15"]
d = up.load(split)
d["ctr"] = d.clicks / d.impressions
d["cell"] = up.cell_key(d)

# ---- headline representations (per unique headline)
H = pd.Series(d.headline.astype(str).unique())
hid = pd.Index(H).get_indexer(d.headline.astype(str))
F15 = up.headline_features(H)
F15 = ((F15 - F15.mean()) / F15.std()).to_numpy()
glove = {}
vocab = set(w for h in H for w in up.tokens(h))
with gzip.open(ROOT / "data" / "external" / "glove" / "glove-wiki-gigaword-100.gz", "rt", encoding="utf8") as fh:
    for line in fh:
        w, rest = line.split(" ", 1)
        if w in vocab:
            glove[w] = np.array(rest.split(), dtype=float)
G = np.array([np.mean([glove[w] for w in up.tokens(h) if w in glove] or [np.zeros(100)], 0) for h in H])
G = (G - G.mean(0)) / (G.std(0) + 1e-9)
DENSE = {"H15": F15, "GloVe+H15": np.c_[G, F15]}

# ---- tests, articles, natural pairs
tests = d.groupby("test").agg(t=("t", "min"), week=("test_week", "first"), lede=("lede_clean", "first")).reset_index()
tix = pd.Index(tests.test).get_indexer(d.test)
pk = [np.where(tix == k)[0] for k in range(len(tests))]
day = (tests.t - tests.t.min()).dt.days.to_numpy()
has_lede = tests.lede.to_numpy() != ""
L = TfidfVectorizer(sublinear_tf=True).fit_transform(tests.lede)
ri, rj = [], []
for a0 in range(0, len(tests), 3000):
    M = (L[a0:a0 + 3000] @ L.T).tocoo()
    r, c = M.row + a0, M.col
    keep = (r < c) & (M.data >= 0.8) & has_lede[r] & has_lede[c]
    ri.append(r[keep]), rj.append(c[keep])
ri, rj = np.concatenate(ri), np.concatenate(rj)
article = connected_components(coo_matrix((np.ones(len(ri)), (ri, rj)), shape=(len(tests),) * 2), directed=False)[1]
nat = day[ri] != day[rj]
ri, rj = ri[nat], rj[nat]
print(f"{split}: {len(d)} packages, {len(tests)} tests, {article.max() + 1} articles, {len(ri)} natural pairs", flush=True)

# ---- evaluation pairs within cells
ev_a, ev_b = [], []
for _, idx in d.groupby("cell").indices.items():
    if len(idx) < 2:
        continue
    a, b = np.triu_indices(len(idx), 1)
    keep = hid[idx[a]] != hid[idx[b]]
    ev_a.append(idx[a][keep]), ev_b.append(idx[b][keep])
ev_a, ev_b = np.concatenate(ev_a), np.concatenate(ev_b)
assert (tix[ev_a] == tix[ev_b]).all(), "evaluation pairs must lie within one test"
p1, p2 = d.ctr.to_numpy()[ev_a], d.ctr.to_numpy()[ev_b]
n1, n2 = d.impressions.to_numpy()[ev_a], d.impressions.to_numpy()[ev_b]
pp = (d.clicks.to_numpy()[ev_a] + d.clicks.to_numpy()[ev_b]) / (n1 + n2)
zz = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2) + 1e-12)
ev_sig = np.abs(zz) > 1.96
ev_art = article[tix[ev_a]]
print(f"evaluation pairs: {len(ev_a)}, significant: {ev_sig.sum()}", flush=True)
y = d.logit_ctr.to_numpy()


def one_per_test(test_ids, r):
    return np.array([pk[k][r.integers(0, len(pk[k]))] for k in test_ids])


def design(rep, rows, tf=None):
    return tf[hid[rows]] if rep == "TF-IDF" else DENSE[rep][hid[rows]]


def fit_ridge(X, t, intercept=True):
    return RidgeCV(alphas=ALPHAS, fit_intercept=intercept).fit(X, t)


rows_out = []
SIG = np.where(ev_sig)[0]
CORR = {}   # (rep, train) -> correctness on each significant within-test pair, filled fold by fold
folds = GroupKFold(5).split(np.zeros(len(tests)), groups=article)
for fold, (tr_t, te_t) in enumerate(folds):
    tr_art = set(article[tr_t])
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(H[np.unique(hid[np.concatenate([pk[k] for k in tr_t])])])
    TF = tfv.transform(H)
    obs = one_per_test(tr_t, rng)
    wk = tests.week.to_numpy()[tix[obs]]
    y_obs = y[obs]
    y_fe = y_obs - pd.Series(y_obs).groupby(wk).transform("mean").to_numpy()
    topic = MiniBatchKMeans(50, random_state=0, n_init=3).fit_predict(G[hid[obs]])
    y_tp = y_fe - pd.Series(y_fe).groupby(topic).transform("mean").to_numpy()
    inp = np.isin(article[ri], list(tr_art))
    na, nb = ri[inp], rj[inp]
    pa, pb = one_per_test(na, rng), one_per_test(nb, rng)
    first = day[na] <= day[nb]
    sgn = np.where(first, 1.0, -1.0)  # orient pairs early - late so the intercept absorbs drift over time
    cell_tr = np.isin(tix, tr_t)
    ci = d.cell.to_numpy()[cell_tr]
    rows_tr = np.where(cell_tr)[0]
    multi = pd.Series(hid[rows_tr]).groupby(ci).transform("nunique").to_numpy() >= 2
    rows_tr, ci = rows_tr[multi], ci[multi]
    cf = pd.factorize(ci)[0]
    yo = y[rows_tr] - pd.Series(y[rows_tr]).groupby(cf).transform("mean").to_numpy()
    te_rows = np.where(np.isin(tix, te_t))[0]
    te_obs = one_per_test(te_t, rng)
    te_pairs = np.isin(ev_art, list(set(article[te_t])))
    for rep in REPS:
        X = lambda r: design(rep, r, TF)
        models = {}
        models["OBS"] = fit_ridge(X(obs), y_obs)
        models["OBS-FE"] = fit_ridge(X(obs), y_fe)
        Dn = (X(pa) - X(pb)).multiply(sgn[:, None]).tocsr() if rep == "TF-IDF" else (X(pa) - X(pb)) * sgn[:, None]
        models["NAT"] = fit_ridge(Dn, (y[pa] - y[pb]) * sgn)
        models["OBS-FE+TOPIC"] = fit_ridge(X(obs), y_tp)
        stack = sparse.vstack([X(obs), Dn]).tocsr() if rep == "TF-IDF" else np.r_[X(obs), Dn]
        models["MIX (OBS-FE + NAT)"] = fit_ridge(stack, np.r_[y_fe, (y[pa] - y[pb]) * sgn])
        Xo = X(rows_tr)
        if rep == "TF-IDF":
            M = sparse.csr_matrix((np.ones(len(cf)), (cf, np.arange(len(cf)))))
            cnt = np.asarray(M.sum(1)).ravel()
            Xo = (Xo - (sparse.diags(1 / cnt) @ M @ Xo).tocsr()[cf]).tocsr()
        else:
            Xo = Xo - pd.DataFrame(Xo).groupby(cf).transform("mean").to_numpy()
        models["ORACLE"] = fit_ridge(Xo, yo, intercept=False)
        if rep == "GloVe+H15":
            models["OBS-HGB"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X(obs), y_obs)
        for mname, mdl in models.items():
            pred = np.zeros(len(d))
            pred[te_rows] = mdl.predict(X(te_rows))
            rho = spearmanr(pred[te_obs], y[te_obs])[0]
            dp = pred[ev_a[te_pairs]] - pred[ev_b[te_pairs]]
            dy = zz[te_pairs]
            ok = np.sign(dp) == np.sign(dy)
            tie = dp == 0
            acc = np.mean(np.where(tie, 0.5, ok))
            sg = ev_sig[te_pairs]
            acc_sig = np.mean(np.where(tie[sg], 0.5, ok[sg]))
            v = CORR.setdefault((rep, mname), np.full(len(SIG), np.nan))
            m_ = np.isin(ev_art[SIG], list(set(article[te_t])))
            dps = pred[ev_a[SIG[m_]]] - pred[ev_b[SIG[m_]]]
            v[m_] = np.where(dps == 0, 0.5, (np.sign(dps) == np.sign(zz[SIG[m_]])).astype(float))
            # decision value: in each test-set cell with >= 2 headlines, CTR of the model's pick vs cell mean
            sub = d.iloc[te_rows][["cell", "ctr"]].assign(p=pred[te_rows], h=hid[te_rows])
            sub = sub[sub.groupby("cell").h.transform("nunique") >= 2]
            g = sub.groupby("cell")
            mean_ctr = g.ctr.mean()
            pick = sub.loc[g.p.idxmax(), ["cell", "ctr"]].set_index("cell").ctr
            best = g.ctr.max()
            lift = float((pick / mean_ctr - 1).mean() * 100)
            ceil = float((best / mean_ctr - 1).mean() * 100)
            rows_out.append(dict(split=split, fold=fold, rep=rep, train=mname, across_spearman=rho,
                                 within_acc=acc, within_acc_sig=acc_sig, n_pairs=int(te_pairs.sum()),
                                 n_sig=int(sg.sum()), lift_pct=lift, ceiling_pct=ceil))
            print(rows_out[-1], flush=True)

keys = sorted(CORR)
np.savez_compressed(ROOT / "results" / "tables" / f"u2_pairs_{split}.npz",
                    configs=np.array(["|".join(c) for c in keys]), correct=np.array([CORR[c] for c in keys]),
                    article=ev_art[SIG])
R = pd.DataFrame(rows_out)
R.to_csv(ROOT / "results" / "tables" / f"u2_ranking_{split}.csv", index=False)
S = R.groupby(["rep", "train"])[["across_spearman", "within_acc", "within_acc_sig", "lift_pct", "ceiling_pct"]].mean()
print(S.round(3).to_string())
