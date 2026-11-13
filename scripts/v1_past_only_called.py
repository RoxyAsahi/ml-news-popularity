"""v19 headline numbers for Mashable under a past-only protocol: what can be called before publication?

Every score is produced by a monthly rolling origin (newspop.anatomy.past_only_scores logic): for each month
m >= 2013-07, fit on all articles published before m and score the articles of m. Evaluation covers the
articles of 2013-07 .. 2014-12. Intervals: bootstrap over publication days (1,000 draws, same draws for every
model, so differences are paired).

Scorers
  hgb         HistGradientBoosting on the 58 dataset features (target log1p shares)
  hgb_crowd   hgb + ex-ante crowding features from real headlines (similar earlier articles, previous 7/30 days)
  hgb_bge     hgb + 32 PCA components of bge headline vectors (PCA fit on the training months only)
  tfidf       ridge on TF-IDF of the real headline
  bge         ridge on bge headline vectors
  kw_avg_avg  the single feature, no fitting

Metrics on the evaluation months
  auc          ROC-AUC for popular (shares >= 1400)
  rho_day      mean over days of the Spearman correlation between score and log shares within the day
  top10        share of days where the top-scored article lands in that day's top 10% by shares
  pair_sec     accuracy on all same-day same-section pairs with different shares (score ties count 1/2)
  pair_day     the same for all same-day pairs
Also: AUC by headline novelty (max TF-IDF cosine to earlier 30 days, quintiles) for the cold-start check.

Output: results/tables/v1_past_only_called.csv (metric x scorer with 95% intervals),
        results/tables/v1_past_only_diffs.csv (paired differences vs hgb),
        results/tables/v1_novelty_auc.csv, data/interim/v1_scores.csv (scores per article).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.stories import channel  # noqa: E402
from newspop.titles import aligned_titles, bge_embeddings  # noqa: E402

FIRST = "2013-07"
B = 1000
OUT = ROOT / "results" / "tables"

df = load().reset_index(drop=True)
n = len(df)
y = np.log1p(df.shares.to_numpy())
pop = (df.shares.to_numpy() >= 1400).astype(int)
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
title = aligned_titles(df).fillna("")
E = bge_embeddings(title)
day = pd.to_datetime(df.publish_date).dt.normalize()
dn = (day - day.min()).dt.days.to_numpy()
month = day.dt.to_period("M").astype(str).to_numpy()
chan = channel(df)

# ex-ante crowding: similar headlines among articles of earlier days (TF-IDF fit without labels on all titles)
V = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(title)
crowd = {k: np.zeros(n) for k in ["prev1_30", "prev3_30", "prev7_30", "prev7_50", "maxsim7", "maxsim30"]}
o = np.argsort(dn, kind="stable"); dso = dn[o]
for i in range(n):
    lo, hi = np.searchsorted(dso, dn[i] - 30), np.searchsorted(dso, dn[i])  # earlier days only
    cand = o[lo:hi]
    if len(cand) == 0:
        continue
    s = (V[cand] @ V[i].T).toarray().ravel(); gap = dn[i] - dn[cand]; p7 = gap <= 7
    crowd["maxsim30"][i] = s.max(); crowd["maxsim7"][i] = s[p7].max() if p7.any() else 0
    crowd["prev1_30"][i] = np.sum((s >= .3) & (gap <= 1)); crowd["prev3_30"][i] = np.sum((s >= .3) & (gap <= 3))
    crowd["prev7_30"][i] = np.sum((s >= .3) & p7); crowd["prev7_50"][i] = np.sum((s >= .5) & p7)
C = pd.DataFrame(crowd)
print(f"articles with a similar (cos >= 0.3) headline in the previous 7 days: {np.mean(C.prev7_30 > 0):.3f}", flush=True)


def hgb():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


scores = {k: np.full(n, np.nan) for k in ["hgb", "hgb_crowd", "hgb_bge", "tfidf", "bge"]}
for m in sorted(set(month)):
    if m < FIRST:
        continue
    tr, te = month < m, month == m
    scores["hgb"][te] = hgb().fit(X[tr], y[tr]).predict(X[te])
    XC = pd.concat([X, C], axis=1)
    scores["hgb_crowd"][te] = hgb().fit(XC[tr], y[tr]).predict(XC[te])
    pca = PCA(32, random_state=0).fit(E[tr])
    XE = np.c_[X.to_numpy(), pca.transform(E)]
    scores["hgb_bge"][te] = hgb().fit(XE[tr], y[tr]).predict(XE[te])
    v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(title[tr])
    scores["tfidf"][te] = Ridge(alpha=10).fit(v.transform(title[tr]), y[tr]).predict(v.transform(title[te]))
    sc = StandardScaler().fit(E[tr])
    scores["bge"][te] = Ridge(alpha=1000).fit(sc.transform(E[tr]), y[tr]).predict(sc.transform(E[te]))
    print(m, "done", flush=True)
scores["ens"] = (scores["hgb"] + scores["bge"]) / 2   # late fusion: mean of the two past-only predictions of log shares
scores["kw_avg_avg"] = df.kw_avg_avg.to_numpy().astype(float)

ev = month >= FIRST
S = pd.DataFrame({k: v[ev] for k, v in scores.items()})
S["y"], S["pop"], S["day"], S["chan"], S["novelty"] = y[ev], pop[ev], dn[ev], chan[ev], 1 - C.maxsim30.to_numpy()[ev]
S["has_title"] = (title[ev] != "").to_numpy()
S.insert(0, "row", np.flatnonzero(ev))
(ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
S.to_csv(ROOT / "data" / "interim" / "v1_scores.csv", index=False)
MODELS = ["hgb", "hgb_crowd", "hgb_bge", "ens", "tfidf", "bge", "kw_avg_avg"]
S = S[S.has_title].reset_index(drop=True)  # articles without a headline (no URL in the file, or slug not in the sitemap) dropped everywhere
days = np.sort(S.day.unique()); di = np.searchsorted(days, S.day.to_numpy()); D = len(days)


def pair_counts(s, yy, g):
    """Per-row-group concordant (ties 1/2) and total pairs with different y, summed within each group."""
    c = t = 0.0
    for idx in g:
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        dy = np.sign(yy[idx[a]] - yy[idx[b]]); ds = np.sign(s[idx[a]] - s[idx[b]]); k = dy != 0
        c += np.sum((dy[k] == ds[k]) + 0.5 * (ds[k] == 0)); t += k.sum()
    return c, t


# per-day statistics: rho, top-pick hit, pair counts (section, day); pooled AUC handled by row resampling
day_rows = [np.flatnonzero(di == d) for d in range(D)]
per = {}
yv = S.y.to_numpy()
for mname in MODELS:
    s = S[mname].to_numpy()
    rho, top, csec, tsec, cday, tday = (np.full(D, np.nan) for _ in range(6))
    for d, idx in enumerate(day_rows):
        if len(idx) >= 10:
            rho[d] = np.corrcoef(rankdata(s[idx]), rankdata(yv[idx]))[0, 1]
            thr = np.quantile(yv[idx], 0.9); top[d] = float(yv[idx][np.argmax(s[idx])] >= thr)
        ch = S.chan.to_numpy()[idx]
        csec[d], tsec[d] = pair_counts(s, yv, [idx[ch == c] for c in np.unique(ch)])
        cday[d], tday[d] = pair_counts(s, yv, [idx])
    per[mname] = dict(rho=rho, top=top, csec=csec, tsec=tsec, cday=cday, tday=tday)
    print(mname, "per-day stats done", flush=True)
base_top = np.nanmean([np.mean(yv[idx] >= np.quantile(yv[idx], 0.9)) if len(idx) >= 10 else np.nan for idx in day_rows])

rng = np.random.default_rng(0)
draws = [rng.integers(0, D, D) for _ in range(B)]
popv = S["pop"].to_numpy()


def stats(mname, w):
    """Metrics given day weights w (bootstrap counts)."""
    p = per[mname]; s = S[mname].to_numpy()
    ok = ~np.isnan(p["rho"])
    rw = w[di]
    auc = roc_auc_score(popv, s, sample_weight=rw) if rw.sum() > 0 else np.nan
    return dict(auc=auc,
                rho_day=np.sum(w[ok] * p["rho"][ok]) / np.sum(w[ok]),
                top10=np.sum(w[ok] * p["top"][ok]) / np.sum(w[ok]),
                pair_sec=np.sum(w * p["csec"]) / np.sum(w * p["tsec"]),
                pair_day=np.sum(w * p["cday"]) / np.sum(w * p["tday"]))


point = {m: stats(m, np.ones(D)) for m in MODELS}
boot = {m: [] for m in MODELS}
for b, dr in enumerate(draws):
    w = np.bincount(dr, minlength=D).astype(float)
    for m in MODELS:
        boot[m].append(stats(m, w))
rows, diffs = [], []
for m in MODELS:
    bm = pd.DataFrame(boot[m])
    for k, v in point[m].items():
        rows.append(dict(scorer=m, metric=k, est=v, lo=bm[k].quantile(.025), hi=bm[k].quantile(.975)))
    if m != "hgb":
        bd = bm - pd.DataFrame(boot["hgb"])
        for k in point[m]:
            diffs.append(dict(scorer=m, metric=k, diff_vs_hgb=point[m][k] - point["hgb"][k],
                              lo=bd[k].quantile(.025), hi=bd[k].quantile(.975)))
R = pd.DataFrame(rows); R.loc[len(R)] = dict(scorer="random", metric="top10", est=base_top, lo=np.nan, hi=np.nan)
R.to_csv(OUT / "v1_past_only_called.csv", index=False)
pd.DataFrame(diffs).to_csv(OUT / "v1_past_only_diffs.csv", index=False)

# cold start: AUC by headline novelty quintile (quintile 5 = least similar to anything in the previous 30 days)
q = pd.qcut(S.novelty.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
nov = []
for k in [1, 2, 3, 4, 5]:
    sel = (q == k).to_numpy()
    nov.append(dict(quintile=k, n=int(sel.sum()), **{m: roc_auc_score(popv[sel], S[m].to_numpy()[sel]) for m in ["hgb", "hgb_bge", "ens", "bge", "tfidf"]}))
pd.DataFrame(nov).to_csv(OUT / "v1_novelty_auc.csv", index=False)

pd.set_option("display.width", 200)
print(R.pivot(index="metric", columns="scorer", values="est").round(4))
print(R.round(4).to_string(index=False))
print(pd.DataFrame(diffs).round(4).to_string(index=False))
print(pd.DataFrame(nov).round(4).to_string(index=False))
