"""Decision-level metrics with a bootstrap over publication days (same definitions as scripts/v1_past_only_called.py).

evaluate(S, models, base) takes a frame with one row per scored article (columns: the model scores, y = log shares,
pop, day, chan) and returns (metrics, paired differences vs each base model):
  auc        ROC-AUC for popular
  rho_day    mean within-day Spearman (days with at least 10 articles)
  top10      share of those days whose top-scored article is in the day's top 10% by shares
  pair_sec   accuracy on same-day same-section pairs with different shares (score ties count 1/2)
  pair_day   the same for all same-day pairs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _pairs(s, yy, groups):
    c = t = 0.0
    for idx in groups:
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        dy = np.sign(yy[idx[a]] - yy[idx[b]]); ds = np.sign(s[idx[a]] - s[idx[b]]); k = dy != 0
        c += np.sum((dy[k] == ds[k]) + 0.5 * (ds[k] == 0)); t += k.sum()
    return c, t


def evaluate(S: pd.DataFrame, models: list[str], base: list[str] = ("hgb",), B: int = 1000, seed: int = 0):
    S = S.reset_index(drop=True)
    days = np.sort(S.day.unique()); di = np.searchsorted(days, S.day.to_numpy()); D = len(days)
    day_rows = [np.flatnonzero(di == d) for d in range(D)]
    yv, popv, chan = S.y.to_numpy(), S["pop"].to_numpy(), S.chan.to_numpy()
    per = {}
    for m in models:
        s = S[m].to_numpy()
        rho, top, csec, tsec, cday, tday = (np.full(D, np.nan) for _ in range(6))
        for d, idx in enumerate(day_rows):
            if len(idx) >= 10:
                rho[d] = np.corrcoef(rankdata(s[idx]), rankdata(yv[idx]))[0, 1]
                top[d] = float(yv[idx][np.argmax(s[idx])] >= np.quantile(yv[idx], 0.9))
            ch = chan[idx]
            csec[d], tsec[d] = _pairs(s, yv, [idx[ch == c] for c in np.unique(ch)])
            cday[d], tday[d] = _pairs(s, yv, [idx])
        per[m] = dict(rho=rho, top=top, csec=csec, tsec=tsec, cday=cday, tday=tday)

    def stats(m, w):
        p = per[m]; ok = ~np.isnan(p["rho"])
        return dict(auc=roc_auc_score(popv, S[m].to_numpy(), sample_weight=w[di]),
                    rho_day=np.sum(w[ok] * p["rho"][ok]) / np.sum(w[ok]),
                    top10=np.sum(w[ok] * p["top"][ok]) / np.sum(w[ok]),
                    pair_sec=np.sum(w * p["csec"]) / np.sum(w * p["tsec"]),
                    pair_day=np.sum(w * p["cday"]) / np.sum(w * p["tday"]))

    rng = np.random.default_rng(seed)
    point = {m: stats(m, np.ones(D)) for m in models}
    boot = {m: [] for m in models}
    for _ in range(B):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        for m in models:
            boot[m].append(stats(m, w))
    boot = {m: pd.DataFrame(v) for m, v in boot.items()}
    rows, diffs = [], []
    for m in models:
        for k, v in point[m].items():
            rows.append(dict(scorer=m, metric=k, est=v, lo=boot[m][k].quantile(.025), hi=boot[m][k].quantile(.975)))
        for b in base:
            if b != m:
                bd = boot[m] - boot[b]
                for k in point[m]:
                    diffs.append(dict(scorer=m, minus=b, metric=k, diff=point[m][k] - point[b][k],
                                      lo=bd[k].quantile(.025), hi=bd[k].quantile(.975)))
    return pd.DataFrame(rows), pd.DataFrame(diffs)
