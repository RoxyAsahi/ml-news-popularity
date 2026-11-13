"""Shared pieces for the editorial-lever analysis (explore/editorial_dml).

Question: which things an editor can still change before publishing (images, videos, length, links,
headline length, weekend, how crowded the day is) move shares once the article's topic and context are
held fixed? Method: partially linear double machine learning (Robinson 1988; Chernozhukov et al. 2018),

    Y = theta * T + g(W) + e,      T = m(W) + v,

with g and m learned by gradient boosting and cross-fitted on folds made of whole weeks, so that the
day- and week-level shocks shared by neighbouring articles never sit on both sides of a split.

Everything expensive (out-of-fold nuisance predictions) is cached in data/interim/dml/, so each script
can be killed and restarted.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from newspop import stories as st
from newspop.data import ROOT, feature_groups, load_raw

OUT = ROOT / "explore" / "editorial_dml" / "out"
CACHE = ROOT / "data" / "interim" / "dml"
# file_row -> true url / headline: the url column of the raw CSV is shifted against its feature rows from June 2013
REALIGNED = ROOT / "data" / "external" / "realigned.csv"
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("THREADS", 4))
N_FOLDS = 5
N_PCA = 64

# ------------------------------------------------------------------ levers (treatments)
# name -> (label, kind). "log" levers are log(1 + count): theta * ln 2 is the effect of doubling (1 + count).
LEVERS = {
    "imgs": ("images", "log"),
    "videos": ("videos", "log"),
    "length": ("body length (words)", "log"),
    "title_len": ("headline length (words)", "log"),
    "links_out": ("links to other sites", "log"),
    "links_self": ("links to Mashable articles", "log"),
    "weekend": ("published at the weekend", "binary"),
    "vol_same": ("same-section articles that day", "log"),
    "vol_other": ("other-section articles that day", "log"),
}
# wording / style: measured by the 2013 dictionary pipeline; "changeable" only in a loose sense
STYLE = {
    "subj_body": ("body subjectivity", "global_subjectivity"),
    "subj_title": ("headline subjectivity", "title_subjectivity"),
    "pol_title": ("headline polarity (abs)", "abs_title_sentiment_polarity"),
    "pos_words": ("share of positive words", "global_rate_positive_words"),
    "neg_words": ("share of negative words", "global_rate_negative_words"),
    "word_len": ("average word length", "average_token_length"),
}
DAY_LEVEL = ["weekend", "vol_same", "vol_other"]
ADDED = ["day", "week", "chan", "dow", "vol_same", "vol_other", "row", "vol_same_lead7", "url_in_file", "title",
         "vol_same_lag1", "vol_same_lead1", "vol_other_lag1", "vol_other_lead1"]  # columns build() adds


def groups(df):
    """feature_groups of the dataset's own columns (without the helper columns build() adds)."""
    return feature_groups(df.drop(columns=[c for c in ADDED if c in df.columns]))


def build():
    """Returns (df, T, S, Y): the analysis sample, lever matrix, style matrix, log shares.

    Rows are taken in raw-file order. Publication date = 2015-01-08 minus `timedelta` (it agrees with the
    dataset's weekday flags on every row; the date inside the file's url column does not, see REALIGNED).
    `url` is replaced by the realigned true url, so newspop.stories works on the right slugs.
    """
    full = load_raw()
    R = pd.read_csv(REALIGNED)
    assert len(R) == len(full) and (R.file_row.to_numpy() == np.arange(len(full))).all()
    full["publish_date"] = pd.Timestamp("2015-01-08") - pd.to_timedelta(full.timedelta, unit="D")
    full["url_in_file"], full["url"], full["title"] = full.url, R.true_url, R.true_title
    chan = st.channel(full)
    day = st.day_index(full)
    d = pd.DataFrame({"day": day, "chan": chan})
    n_day = d.groupby("day").day.transform("size")
    n_same = d.groupby(["day", "chan"]).day.transform("size")
    full = full.assign(day=day, week=day // 7, chan=chan, dow=full.publish_date.dt.day_of_week,
                       vol_same=np.log(n_same.to_numpy()),
                       vol_other=np.log((n_day - n_same).clip(lower=1).to_numpy()),
                       row=np.arange(len(full)))
    # volume one week later (same weekday): placebo treatment
    key = d.assign(n=n_same).drop_duplicates(["day", "chan"]).set_index(["day", "chan"]).n
    full["vol_same_lead7"] = np.log(pd.MultiIndex.from_arrays([day + 7, chan]).map(key).to_numpy(dtype=float))
    nday = d.groupby("day").size()
    for nme, sh in [("lag1", -1), ("lead1", 1)]:     # neighbouring days, for the spillover check (d7)
        ns = pd.MultiIndex.from_arrays([day + sh, chan]).map(key).to_numpy(dtype=float)
        nd = pd.Series(day + sh).map(nday).to_numpy(dtype=float)
        full[f"vol_same_{nme}"] = np.log(ns)
        full[f"vol_other_{nme}"] = np.log(np.clip(nd - ns, 1, None))
    keep = ((full.n_tokens_content > 0)      # 1,181 pages whose text was not parsed: all content features are 0
            & (full.n_unique_tokens <= 1)    # one corrupt row
            & full.title.notna()).to_numpy()  # last 152 rows (true url outside the file) + 24 without a sitemap headline
    df = full[keep].reset_index(drop=True)
    T = pd.DataFrame({
        "imgs": np.log1p(df.num_imgs), "videos": np.log1p(df.num_videos),
        "length": np.log1p(df.n_tokens_content), "title_len": np.log1p(df.n_tokens_title),
        "links_out": np.log1p((df.num_hrefs - df.num_self_hrefs).clip(lower=0)),
        "links_self": np.log1p(df.num_self_hrefs),
        "weekend": df.is_weekend.astype(float), "vol_same": df.vol_same, "vol_other": df.vol_other})
    S = pd.DataFrame({k: df[c] for k, (_, c) in STYLE.items()})
    Y = np.log1p(df.shares.to_numpy())
    return df, T, S, Y


def title_pca(df, n=N_PCA):
    """First n principal components of the bge headline vectors (unsupervised; no shares involved)."""
    f = CACHE / f"title_pca{n}.npy"
    if f.exists():
        return np.load(f)
    from sklearn.decomposition import PCA
    E = np.load(CACHE / "title_bge_true.npy")[df.row.to_numpy()]
    assert not np.isnan(E).any(), "run d0_title_embeddings.py first"
    P = PCA(n, random_state=0).fit_transform(E - E.mean(0)).astype(np.float32)
    np.save(f, P)
    return P


def control_sets(df):
    """Named confounder sets W (the 'ladder'). Each is a 2-D float array; trees do the rest."""
    g = groups(df)
    chan_d = pd.get_dummies(pd.Series(df.chan), dtype=float).to_numpy()
    ctx_nodow = np.column_stack([df.day, chan_d])
    ctx = np.column_stack([ctx_nodow, df.dow])
    kw = df[g["keyword"]].to_numpy(float)
    lda = df[g["lda"]].to_numpy(float)
    nk = df[["num_keywords"]].to_numpy(float)
    P = title_pca(df)
    selfref = np.log1p(df[g["self_reference"]].to_numpy(float))
    style = df[g["sentiment"] + ["average_token_length", "n_unique_tokens", "n_non_stop_unique_tokens"]].to_numpy(float)
    W = {}
    for tag, c in [("", ctx), ("_nodow", ctx_nodow)]:
        W["ctx" + tag] = c
        W["topic" + tag] = np.column_stack([c, lda, kw, nk])
        W["title" + tag] = np.column_stack([c, lda, kw, nk, P])           # main specification
        W["ref" + tag] = np.column_stack([c, lda, kw, nk, P, selfref])    # + shares of the Mashable articles it links to
        W["ext" + tag] = np.column_stack([c, lda, kw, nk, P, selfref, style])
        # leave-one-group-out versions of the main set, for sensitivity benchmarks
        W["title-kw" + tag] = np.column_stack([c, lda, nk, P])
        W["title-ldachan" + tag] = np.column_stack([c[:, [0] + ([c.shape[1] - 1] if tag == "" else [])], kw, nk, P])
        W["title-emb" + tag] = W["topic" + tag]
        W["title-time" + tag] = np.column_stack([c[:, 1:], lda, kw, nk, P])
    W["content"] = np.column_stack([chan_d, lda, kw, nk, P])          # what the article is, no date or weekday (d7)
    return W


def wname(base, lever):
    """Weekend cannot be compared holding the weekday fixed, so it uses the sets without day-of-week."""
    return base + "_nodow" if lever == "weekend" else base


def week_folds(df, rep, k=N_FOLDS):
    weeks = np.unique(df.week)
    rng = np.random.default_rng(1000 + rep)
    assign = dict(zip(weeks, rng.permutation(len(weeks)) % k))
    return df.week.map(assign).to_numpy()


def random_folds(df, rep, k=N_FOLDS):
    return np.random.default_rng(2000 + rep).permutation(len(df)) % k


def make_learner(binary=False, seed=0):
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_jobs=THREADS, random_state=seed, verbose=-1)
    return lgb.LGBMClassifier(**p) if binary else lgb.LGBMRegressor(**p)


def oof(name, target, X, folds, tag, binary=False):
    """Out-of-fold prediction of `target` from X; cached under CACHE/<tag>__<name>.npy."""
    f = CACHE / f"{tag}__{name}.npy"
    if f.exists():
        return np.load(f)
    pred = np.zeros(len(target))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = make_learner(binary, seed=int(k)).fit(X[tr], target[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if binary else m.predict(X[te])
    np.save(f, pred)
    return pred


def residuals(df, name, target, base, rep, W, lever=None, folds="week"):
    """target - E[target | W_base], cross-fitted. `lever` only decides whether day-of-week is in W."""
    wn = wname(base, lever) if lever else base
    fo = week_folds(df, rep) if folds == "week" else random_folds(df, rep)
    binary = name == "weekend"
    return target - oof(name, target, W[wn], fo, f"{wn}__{folds}__r{rep}", binary)


# ------------------------------------------------------------------ final stage
def cluster_ols(y, X, groups):
    """OLS without intercept on residualised data, with cluster-robust (CR1) covariance."""
    X = np.asarray(X, float).reshape(len(y), -1)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g = pd.factorize(groups)[0]
    S = np.zeros((g.max() + 1, X.shape[1]))
    np.add.at(S, g, X * u[:, None])
    G, n, k = S.shape[0], len(y), X.shape[1]
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, V


def combine(thetas, ses):
    """Median over cross-fitting repetitions (Chernozhukov et al. 2018, eq. 3.14)."""
    thetas, ses = np.asarray(thetas), np.asarray(ses)
    med = np.median(thetas, 0)
    se = np.sqrt(np.median(ses ** 2 + (thetas - med) ** 2, 0))
    return med, se


def pct(theta, kind):
    """Effect in % of shares: doubling (1 + count) for log levers; 0 -> 1 for binary ones."""
    return 100 * (np.exp(theta * (np.log(2) if kind == "log" else 1.0)) - 1)
