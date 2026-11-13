"""Probe B: does the training target matter when the decision is about total shares (heavy tail)?

Past-only monthly rolling origin (fit on months < m, score month m, m >= 2013-07), 58 dataset features.
Scorers
  log_mean   HGB squared error on log1p(shares)                (what v19 and most of the literature do)
  poisson    HGB Poisson deviance on raw shares                (estimates E[shares] directly)
  gamma      HGB Gamma deviance on raw shares
  lognorm    exp(mu + s2/2) with s2 from a second HGB fit on out-of-fold squared log residuals
  q90        HGB pinball loss, 0.9 quantile of log shares
  clf_top5   HGB classifier for shares >= training 95th percentile
  clf_1400   HGB classifier for shares >= 1400
Metrics (evaluation months), all per day then pooled
  auc1400, auc_top5 (threshold = pooled 95th pct), rho_day,
  cap@k      share of ALL shares captured by picking each day's top-k scored articles, k = 1, 3, 10%
  hit1@k     share of top-1% articles (pooled threshold) among the picks
Resumable: per-month scores cached in data/interim/n1_scores.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402

FIRST = "2013-07"
CACHE = ROOT / "data" / "interim" / "n1_scores.parquet"
df = load().reset_index(drop=True)
n = len(df)
s = df.shares.to_numpy().astype(float)
y = np.log1p(s)
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number").to_numpy()
day = pd.to_datetime(df.publish_date).dt.normalize()
dn = (day - day.min()).dt.days.to_numpy()
month = day.dt.to_period("M").astype(str).to_numpy()
NAMES = ["log_mean", "poisson", "gamma", "lognorm", "sig2", "q90", "clf_top5", "clf_1400"]


def reg(**kw):
    return HistGradientBoostingRegressor(random_state=0, **{**HGB_PARAMS, **kw})


if CACHE.exists():
    S = pd.read_parquet(CACHE)
else:
    S = pd.DataFrame({k: np.full(n, np.nan) for k in NAMES})
for m in sorted(set(month)):
    if m < FIRST or not np.isnan(S.loc[month == m, "clf_1400"]).any():
        continue
    tr, te = month < m, month == m
    Xtr, ytr, str_ = X[tr], y[tr], s[tr]
    f = reg().fit(Xtr, ytr); mu = f.predict(X[te]); S.loc[te, "log_mean"] = mu
    S.loc[te, "poisson"] = reg(loss="poisson").fit(Xtr, str_).predict(X[te])
    S.loc[te, "gamma"] = reg(loss="gamma").fit(Xtr, str_).predict(X[te])
    oof = np.zeros(tr.sum())
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xtr):
        oof[b] = reg().fit(Xtr[a], ytr[a]).predict(Xtr[b])
    g = reg(loss="gamma", max_iter=200).fit(Xtr, (ytr - oof) ** 2 + 1e-6)
    s2 = g.predict(X[te]); S.loc[te, "sig2"] = s2; S.loc[te, "lognorm"] = mu + s2 / 2
    S.loc[te, "q90"] = reg(loss="quantile", quantile=0.9).fit(Xtr, ytr).predict(X[te])
    thr = np.quantile(str_, 0.95)
    c = HistGradientBoostingClassifier(random_state=0, **HGB_PARAMS)
    S.loc[te, "clf_top5"] = c.fit(Xtr, str_ >= thr).predict_proba(X[te])[:, 1]
    c = HistGradientBoostingClassifier(random_state=0, **HGB_PARAMS)
    S.loc[te, "clf_1400"] = c.fit(Xtr, str_ >= 1400).predict_proba(X[te])[:, 1]
    CACHE.parent.mkdir(parents=True, exist_ok=True); S.to_parquet(CACHE)
    print(m, "done", flush=True)

ev = month >= FIRST
E = S[ev].reset_index(drop=True); se, de = s[ev], dn[ev]
t95, t99 = np.quantile(se, 0.95), np.quantile(se, 0.99)
days = np.unique(de); groups = [np.flatnonzero(de == d) for d in days]
rng = np.random.default_rng(0)


def day_stats(score):
    """Per-day arrays: captured shares and top-1% hits for k=1,3,10%; within-day Spearman."""
    out = np.zeros((len(groups), 8))
    for gi, ix in enumerate(groups):
        o = ix[np.argsort(-score[ix], kind="stable")]
        k10 = max(1, int(round(0.1 * len(ix))))
        for j, k in enumerate([1, 3, k10]):
            out[gi, j] = se[o[:k]].sum(); out[gi, 3 + j] = (se[o[:k]] >= t99).sum()
        out[gi, 6] = se[ix].sum()
        r1, r2 = rankdata(score[ix]), rankdata(se[ix])
        out[gi, 7] = np.corrcoef(r1, r2)[0, 1] if len(ix) > 2 and r1.std() > 0 and r2.std() > 0 else np.nan
    return out


rows = []
B = 1000
boot = rng.integers(0, len(groups), size=(B, len(groups)))
base = None
allstats = {}
for k in ["log_mean", "poisson", "gamma", "lognorm", "q90", "clf_top5", "clf_1400", "random", "oracle"]:
    sc = rng.random(ev.sum()) if k == "random" else se if k == "oracle" else E[k].to_numpy()
    st = day_stats(sc); allstats[k] = st
    tot = st[:, 6].sum()
    r = dict(scorer=k, auc1400=roc_auc_score(se >= 1400, sc), auc_top5=roc_auc_score(se >= t95, sc),
             auc_top1=roc_auc_score(se >= t99, sc), rho_day=np.nanmean(st[:, 7]))
    for j, nm in enumerate(["cap@1", "cap@3", "cap@10%"]):
        r[nm] = st[:, j].sum() / tot
        bs = st[boot][:, :, j].sum(1) / st[boot][:, :, 6].sum(1)
        r[nm + "_lo"], r[nm + "_hi"] = np.quantile(bs, [.025, .975])
    for j, nm in enumerate(["hit1@1", "hit1@3", "hit1@10%"]):
        r[nm] = st[:, 3 + j].sum() / (se >= t99).sum()
    rows.append(r)
R = pd.DataFrame(rows).set_index("scorer")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(3).to_string())
# paired differences vs log_mean for capture
for k in ["poisson", "gamma", "lognorm", "q90", "clf_top5"]:
    for j, nm in enumerate(["cap@1", "cap@3", "cap@10%"]):
        a, b = allstats[k], allstats["log_mean"]
        d = a[boot][:, :, j].sum(1) / a[boot][:, :, 6].sum(1) - b[boot][:, :, j].sum(1) / b[boot][:, :, 6].sum(1)
        print(f"{k:9s} {nm:8s} diff vs log_mean {d.mean():+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")
# is variance predictable? calibration of sig2 by quintile
res2 = (y[ev] - E.log_mean.to_numpy()) ** 2
q = pd.qcut(E.sig2, 5, labels=False)
print("predicted sig2 quintile -> mean predicted, realised squared residual")
print(pd.DataFrame({"pred": E.sig2, "real": res2}).groupby(q).mean().round(3).to_string())
print("n eval", ev.sum(), "days", len(groups), "t95", t95, "t99", t99)
(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
R.to_csv(ROOT / "results" / "tables" / "n1_tail_value.csv")
