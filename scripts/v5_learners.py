"""Model families under the past-only protocol (repaired data).

Regressors of log shares on the 58 features (ridge, k-nearest neighbours, random forest, HistGB) and on the real
headline (ridge on TF-IDF, ridge on bge embeddings). Hyper-parameters are chosen once, by accuracy on same-day
same-section pairs in March-June 2013 under the same rolling origin (train on all earlier months), then frozen
and scored on 2013-07 to 2014-12, the months of the main table. HistGB keeps the setting used everywhere else.
95% intervals: bootstrap over days (1,000 draws).
Output: results/tables/v5_learners.csv, v5_learners_selection.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.stories import channel  # noqa: E402
from newspop.titles import aligned_titles, bge_embeddings  # noqa: E402

T = ROOT / "results" / "tables"; B = 1000; rng = np.random.default_rng(0)
df = load().reset_index(drop=True); n = len(df)
y = np.log1p(df.shares.to_numpy()); pop = (df.shares.to_numpy() >= 1400).astype(int)
X = df.drop(columns=[c for c in ["url", "url_in_file", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number").to_numpy()
title = aligned_titles(df).fillna(""); ok = (title != "").to_numpy(); E = bge_embeddings(title)
date = pd.to_datetime(df.publish_date).dt.normalize(); dn = (date - date.min()).dt.days.to_numpy()
month = date.dt.to_period("M").astype(str).to_numpy(); chan = np.asarray(channel(df))


def feat(make):
    return lambda tr, te: make().fit(X[tr], y[tr]).predict(X[te])


def tfidf(alpha):
    def f(tr, te):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(title[tr])
        return Ridge(alpha=alpha).fit(v.transform(title[tr]), y[tr]).predict(v.transform(title[te]))
    return f


def bge(alpha):
    def f(tr, te):
        sc = StandardScaler().fit(E[tr])
        return Ridge(alpha=alpha).fit(sc.transform(E[tr]), y[tr]).predict(sc.transform(E[te]))
    return f


FAMILIES = {
    "ridge, 58 features": {a: feat(lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (1, 100, 1000, 10000)},
    "k-NN, 58 features": {k: feat(lambda k=k: make_pipeline(StandardScaler(), KNeighborsRegressor(k))) for k in (50, 150, 500)},
    "random forest, 58 features": {m: feat(lambda m=m: RandomForestRegressor(200, min_samples_leaf=m, max_features=0.33, n_jobs=2, random_state=0)) for m in (5, 20)},
    "HistGB, 58 features": {"fixed": feat(lambda: HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS))},
    "ridge, headline TF-IDF": {a: tfidf(a) for a in (1, 10, 100)},
    "ridge, headline embedding": {a: bge(a) for a in (100, 1000, 10000)},
}


def rolling(f, months):
    s = np.full(n, np.nan)
    for m in months:
        tr, te = np.flatnonzero((month < m) & ok), np.flatnonzero((month == m) & ok)
        s[te] = f(tr, te)
    return s


def day_stats(s, idx):
    """Per day: concordant and total same-day same-section pairs, within-day Spearman."""
    out = []
    for d, g in pd.DataFrame({"i": idx, "d": dn[idx]}).groupby("d").i:
        ix = g.to_numpy(); c = t = 0.0
        for ch in np.unique(chan[ix]):
            jx = ix[chan[ix] == ch]
            if len(jx) > 1:
                a, b = np.triu_indices(len(jx), 1)
                dy = np.sign(y[jx[a]] - y[jx[b]]); ds = np.sign(s[jx[a]] - s[jx[b]]); k = dy != 0
                c += np.sum(dy[k] == ds[k]) + 0.5 * np.sum(ds[k] == 0); t += k.sum()
        rho = np.corrcoef(rankdata(s[ix]), rankdata(y[ix]))[0, 1] if len(ix) >= 10 else np.nan
        out.append((c, t, rho))
    return np.array(out)


VAL = [m for m in sorted(set(month)) if "2013-03" <= m <= "2013-06"]
TEST = [m for m in sorted(set(month)) if m >= "2013-07"]
sel_rows, rows = [], []
for fam, grid in FAMILIES.items():
    best = None
    for hp, f in grid.items():
        if len(grid) == 1:
            best = (hp, np.nan); break
        st = day_stats(rolling(f, VAL), np.flatnonzero(np.isin(month, VAL) & ok)); acc = st[:, 0].sum() / st[:, 1].sum()
        sel_rows.append(dict(family=fam, setting=hp, val_pair_sec=acc)); print(fam, hp, round(acc, 4), flush=True)
        if best is None or acc > best[1]:
            best = (hp, acc)
    s = rolling(grid[best[0]], TEST); idx = np.flatnonzero(np.isin(month, TEST) & ok)
    st = day_stats(s, idx); okr = ~np.isnan(st[:, 2])
    days, inv = np.unique(dn[idx], return_inverse=True)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(st), len(st)), minlength=len(st)).astype(float)
        bs.append(((w * st[:, 0]).sum() / (w * st[:, 1]).sum(), (w[okr] * st[okr, 2]).sum() / w[okr].sum(),
                   roc_auc_score(pop[idx], s[idx], sample_weight=w[inv])))
    bs = np.array(bs)
    for j, (metric, est) in enumerate([("pair_sec", st[:, 0].sum() / st[:, 1].sum()), ("rho_day", np.nanmean(st[:, 2])), ("auc", roc_auc_score(pop[idx], s[idx]))]):
        rows.append(dict(family=fam, chosen=best[0], metric=metric, est=est, lo=np.quantile(bs[:, j], .025), hi=np.quantile(bs[:, j], .975)))
    print(fam, "chosen", best[0], {r["metric"]: round(r["est"], 4) for r in rows[-3:]}, flush=True)
pd.DataFrame(sel_rows).to_csv(T / "v5_learners_selection.csv", index=False)
R = pd.DataFrame(rows); R.to_csv(T / "v5_learners.csv", index=False)
print(R.round(4).to_string(index=False))
