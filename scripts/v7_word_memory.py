"""A past-only memory of headline words, as a visible replacement for the dataset's kw_* columns (repaired data).

The nine kw_* columns are statistics of the shares of earlier articles that carry the same editor keyword; the
keywords themselves are not in the data. Here the same kind of memory is rebuilt from keys anyone can see: the
words and bigrams of the real headline (lower case, English stop words removed).

For an article published on day t and a key k, with the articles published on or before day t - L:
    n_k, mean_k   number and mean log shares of the earlier articles whose headline contains k
    mu            mean log shares of all earlier articles
    lift_k = n_k / (n_k + m) * (mean_k - mu)          (empirical-Bayes shrinkage towards no lift)
m = sigma^2 / tau^2 by the method of moments on articles before July 2013 only (tau^2 = mean over article-key pairs
of (y_i - mu)(mean_k - mu), sigma^2 = variance of y_i - mu). Per article: mean, max, min and sum of the lifts of its
keys, the number of keys with a history, and log(1 + largest n_k).
L = 1: all earlier days (what kw_* does). L = 30: only articles at least 30 days old, whose share counts were close
to final by day t. In both cases the share counts are those recorded in January 2015 ("past articles, retrospective
labels"), exactly as for kw_*.

Scores (monthly rolling origin and metrics of v1_past_only_called.py, same rows):
  hgb            58 features (from v1)                     bge, ens      headline ridge and mean fusion (from v1)
  hgb_wm         kw_* replaced by the word memory (L = 1)  hgb_wm30      the same with L = 30
  hgb_nokw       58 features without kw_*                  hgb_all_wm    all 58 features + word memory (L = 1)
  ens_wm, ens_wm30, ens_all_wm    mean of the HistGB score above and the headline ridge
  wm_mean, wm30_mean              the mean lift alone, no fitting (compare kw_avg_avg)
Output: results/tables/v7_word_memory.csv, v7_word_memory_diffs.csv, data/interim/v7_scores.csv
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.dayeval import evaluate  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

FIRST, BURN_END = "2013-07", pd.Timestamp("2013-07-01")
df = load().reset_index(drop=True); n = len(df)
y = np.log1p(df.shares.to_numpy())
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
title = aligned_titles(df).fillna("")
day = pd.to_datetime(df.publish_date).dt.normalize()
dn = (day - day.min()).dt.days.to_numpy(); month = day.dt.to_period("M").astype(str).to_numpy()
burn = (day < BURN_END).to_numpy()
assert np.all(np.diff(dn) >= 0)
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def keys(t):
    tk = [w for w in TOKEN.findall(t.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return sorted(set(tk) | {a + "_" + b for a, b in zip(tk, tk[1:])})


K = [keys(t) for t in title]
print("keys per headline", round(float(np.mean([len(k) for k in K if k])), 1), flush=True)


def memory(L):
    """Per article: list of (n_k, mean_k - mu) over its keys with a history, using articles of days <= t - L."""
    cnt, tot = defaultdict(float), defaultdict(float); N = S = 0.0; j = 0; out, mu = [], np.full(n, np.nan)
    for i in range(n):
        while j < n and dn[j] <= dn[i] - L:   # admit articles that are now at least L days old
            for k in K[j]:
                cnt[k] += 1; tot[k] += y[j]
            N += 1; S += y[j]; j += 1
        if N:
            mu[i] = S / N
        out.append([(cnt[k], tot[k] / cnt[k] - mu[i]) for k in K[i] if cnt.get(k, 0) > 0])
    return out, mu


def features(L):
    mem, mu = memory(L)
    num = den = 0.0; res = []
    for i in np.flatnonzero(burn & ~np.isnan(mu)):
        for _, dev in mem[i]:
            num += (y[i] - mu[i]) * dev; den += 1
        res.append(y[i] - mu[i])
    tau2, sig2 = num / den, float(np.var(res)); m = sig2 / tau2
    print(f"L={L}: tau2 {tau2:.4f} sigma2 {sig2:.3f} m {m:.1f}", flush=True)
    F = np.full((n, 6), np.nan)
    for i, lst in enumerate(mem):
        if lst:
            nk = np.array([a for a, _ in lst]); th = nk / (nk + m) * np.array([d for _, d in lst])
            F[i] = [th.mean(), th.max(), th.min(), th.sum(), len(lst), np.log1p(nk.max())]
        elif K[i]:
            F[i, 4:] = 0
    return pd.DataFrame(F, columns=[f"wm{L}_{c}" for c in ["mean", "max", "min", "sum", "nkeys", "lognmax"]])


W1, W30 = features(1), features(30)
KW = [c for c in X.columns if c.startswith("kw_")]
NOKW = X.drop(columns=KW)
SETS = {"hgb_nokw": NOKW, "hgb_wm": pd.concat([NOKW, W1], axis=1), "hgb_wm30": pd.concat([NOKW, W30], axis=1),
        "hgb_all_wm": pd.concat([X, W1], axis=1)}
scores = {k: np.full(n, np.nan) for k in SETS}
for mth in sorted(set(month)):
    if mth < FIRST:
        continue
    tr, te = month < mth, month == mth
    for k, Z in SETS.items():
        scores[k][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(Z[tr], y[tr]).predict(Z[te])
    print(mth, "done", flush=True)

S = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv")
for k, v in scores.items():
    S[k] = v[S.row]
S["wm_mean"], S["wm30_mean"] = W1.wm1_mean.fillna(0).to_numpy()[S.row], W30.wm30_mean.fillna(0).to_numpy()[S.row]
for k in ["hgb_wm", "hgb_wm30", "hgb_all_wm"]:
    S["ens_" + k[4:]] = (S[k] + S.bge) / 2
S.to_csv(ROOT / "data" / "interim" / "v7_scores.csv", index=False)
MODELS = ["hgb", "hgb_nokw", "hgb_wm", "hgb_wm30", "hgb_all_wm", "bge", "ens", "ens_wm", "ens_wm30", "ens_all_wm",
          "wm_mean", "wm30_mean", "kw_avg_avg"]
R, Dd = evaluate(S[S.has_title], MODELS, base=["hgb", "ens", "kw_avg_avg"])
T = ROOT / "results" / "tables"
R.to_csv(T / "v7_word_memory.csv", index=False); Dd.to_csv(T / "v7_word_memory_diffs.csv", index=False)
pd.set_option("display.width", 220)
print(R.pivot(index="scorer", columns="metric", values="est").loc[MODELS].round(4))
print(Dd[Dd.metric.isin(["auc", "pair_sec"])].round(4).to_string(index=False))
