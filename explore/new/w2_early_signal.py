"""Direction A, first numbers: how much does a look at the first hours add to what is known before publication?

For a horizon h (hours after publication) an article is "covered" if the archive holds at least one sighting of
it with a share count at age <= h. Its early state is the LAST such sighting.

Scorers, all past-only (monthly rolling origin, fit on months < m, score month m, m >= --first):
  prepub   HGB on the 58 dataset features, target log1p(final shares)        (cached by n1_tail_value.py)
  sh       linear on log1p(shares at sighting), age, their interaction       (Szabo-Huberman style)
  early    HGB on the early state only (counts per network, age, column, rank, velocity, growth)
  both     HGB on the 58 features + prepub score + early state
Metrics on the covered articles of the evaluation months (prepub is re-scored on the same articles):
  auc1400, auc_top5, auc_top1 (thresholds pooled over the dataset), spearman, r2 (log scale),
  auc_below: AUC for ending >= 1400 among articles still below 1400 at the sighting (the non-trivial part),
  cap10: share of the covered articles' total shares captured by each day's top 10% picks.
Intervals: bootstrap over publication days.

Output: results/tables/w2_early_signal.csv, results/tables/w2_coverage.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HGB_PARAMS, ROOT, features, load_fixed  # noqa: E402

NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]
ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2013-07")
ap.add_argument("--horizons", default="0.5,1,2,3,6,12,24,48")
ap.add_argument("--obs", default=str(ROOT / "data" / "interim" / "wb_obs.parquet"))
ap.add_argument("--boot", type=int, default=500)
ap.add_argument("--src", default="all", help="all, or lists (exclude article-page captures)")
a = ap.parse_args()

df = load_fixed()
n = len(df)
X = features(df)
y = np.log1p(df.shares.to_numpy().astype(float))
s = df.shares.to_numpy().astype(float)
month = df.date.dt.to_period("M").astype(str).to_numpy()
dn = (df.date - df.date.min()).dt.days.to_numpy()
t95, t99 = np.quantile(s, .95), np.quantile(s, .99)
S = pd.read_parquet(ROOT / "data" / "interim" / "n1_scores_fixed.parquet")
prepub = S.log_mean.to_numpy()
assert len(S) == n

O = pd.read_parquet(a.obs)
if a.src == "lists":
    O = O[O.src != "article"]
O = O[(O.age_h > 0) & O.total.notna()].copy()
row_of = pd.Series(np.arange(n), index=df.art_id.to_numpy())
row_of = row_of[row_of.index.notna() & ~row_of.index.duplicated()]
O["row"] = O.art_id.map(row_of)
print(f"sightings {len(O):,}; on dataset articles {O.row.notna().sum():,}; dataset articles sighted "
      f"{O.row.dropna().nunique():,} of {n:,}", flush=True)
O = O.dropna(subset=["row"]).astype({"row": int}).sort_values(["row", "ts"])
# monotone sanity: early count should not exceed the final count
viol = (O.total > s[O.row.to_numpy()] * 1.05 + 100).mean()
print(f"sightings with early count above the final count: {viol:.3%}")
COLS = ["new", "rising", "hot", "page"]


def early_state(h):
    """Last sighting at age <= h per article, plus growth since the previous sighting."""
    o = O[O.age_h <= h]
    g = o.groupby("row")
    last = g.tail(1).set_index("row")
    E = pd.DataFrame(index=last.index)
    E["age"] = last.age_h
    E["l_total"] = np.log1p(last.total)
    for k in NETS:
        E["l_" + k] = np.log1p(last[k].fillna(0))
    E["fb_share"] = last.facebook.fillna(0) / (last.total + 1)
    for c in COLS:
        E["col_" + c] = (last.col == c).astype(float)
    E["rank"] = last["rank"]
    E["vel_last"], E["vel_slope"], E["vel_n"] = last.vel_last, last.vel_slope, last.vel_n
    E["n_sight"] = g.size()
    E["ever_rising"] = g.col.apply(lambda v: float((v == "rising").any()))
    E["ever_hot"] = g.col.apply(lambda v: float((v == "hot").any()))
    prev = g.nth(-2)
    if len(prev):
        prev = prev.set_index("row")
        dt_ = (last.age_h - prev.age_h.reindex(last.index)).clip(lower=1e-3)
        E["rate"] = (np.log1p(last.total) - np.log1p(prev.total.reindex(last.index))) / dt_
    else:
        E["rate"] = np.nan
    E["l_rate0"] = E.l_total - np.log(E.age.clip(lower=0.05))  # log average shares per hour so far
    return E


def reg():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


rng = np.random.default_rng(0)
rows, cov = [], []
ev_all = month >= a.first
for h in [float(x) for x in a.horizons.split(",")]:
    E = early_state(h)
    idx = E.index.to_numpy()
    covered = np.zeros(n, bool); covered[idx] = True
    for half, mask in [("2013H1", month < "2013-07"), ("2013H2", (month >= "2013-07") & (month < "2014-01")),
                       ("2014H1", (month >= "2014-01") & (month < "2014-07")), ("2014H2", month >= "2014-07")]:
        cov.append(dict(h=h, period=half, n=int(mask.sum()), covered=float(covered[mask].mean()),
                        mean_log_cov=float(y[mask & covered].mean()) if (mask & covered).any() else np.nan,
                        mean_log_unc=float(y[mask & ~covered].mean()) if (mask & ~covered).any() else np.nan))
    Ef = E.to_numpy(dtype=float)
    Ef[:, np.isnan(Ef).all(0)] = 0.0  # HGB cannot bin an all-missing column
    Xi = X.iloc[idx].to_numpy(); yi = y[idx]; mi = month[idx]; pi = prepub[idx]
    sc = {k: np.full(len(idx), np.nan) for k in ["sh", "early", "both"]}
    for m in sorted(set(mi)):
        if m < a.first:
            continue
        tr, te = mi < m, mi == m
        if tr.sum() < 300 or te.sum() == 0:
            continue
        dead = np.isnan(Ef[tr]).all(0)
        if dead.any():
            Ef = Ef.copy(); Ef[:, dead] = np.nan_to_num(Ef[:, dead], nan=0.0)
        L = np.c_[Ef[:, 1], np.log(Ef[:, 0].clip(min=.05)), Ef[:, 1] * np.log(Ef[:, 0].clip(min=.05))]
        sc["sh"][te] = LinearRegression().fit(L[tr], yi[tr]).predict(L[te])
        sc["early"][te] = reg().fit(Ef[tr], yi[tr]).predict(Ef[te])
        # the prepub score is only available from --first on, so "both" stacks features + early state directly
        B = np.c_[Xi, Ef]
        sc["both"][te] = reg().fit(B[tr], yi[tr]).predict(B[te])
    sc["prepub"] = pi
    ok = ~np.isnan(sc["early"]) & ~np.isnan(pi)
    si, di = s[idx][ok], dn[idx][ok]
    cur = np.expm1(Ef[ok, 1])
    days = np.unique(di); gi = [np.flatnonzero(di == d) for d in days]
    boot = rng.integers(0, len(days), size=(a.boot, len(days)))

    def metrics(score, sub):
        out = {}
        yy = np.log1p(si[sub]); z = score[sub]
        for nm, thr in [("auc1400", 1400), ("auc_top5", t95), ("auc_top1", t99)]:
            lab = si[sub] >= thr
            out[nm] = roc_auc_score(lab, z) if 0 < lab.sum() < len(lab) else np.nan
        out["spearman"] = spearmanr(z, yy)[0]
        below = cur[sub] < 1400
        lab = si[sub][below] >= 1400
        out["auc_below"] = roc_auc_score(lab, z[below]) if 0 < lab.sum() < len(lab) else np.nan
        return out

    for k in ["prepub", "sh", "early", "both"]:
        z = sc[k][ok]
        r = dict(h=h, scorer=k, n=int(ok.sum()), **metrics(z, np.arange(ok.sum())))
        r["r2"] = r2_score(np.log1p(si), z)
        capn = np.zeros(len(days)); capd = np.zeros(len(days))
        for j, ix in enumerate(gi):
            o = ix[np.argsort(-z[ix], kind="stable")][:max(1, int(round(.1 * len(ix))))]
            capn[j], capd[j] = si[o].sum(), si[ix].sum()
        r["cap10"] = capn.sum() / capd.sum()
        bs = capn[boot].sum(1) / capd[boot].sum(1)
        r["cap10_lo"], r["cap10_hi"] = np.quantile(bs, [.025, .975])
        aucs = []
        for b in boot[:200]:
            sub = np.concatenate([gi[j] for j in b])
            aucs.append(metrics(z, sub)["auc1400"])
        r["auc1400_lo"], r["auc1400_hi"] = np.nanquantile(aucs, [.025, .975])
        rows.append(r)
    print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)

out = ROOT / "results" / "tables"; out.mkdir(parents=True, exist_ok=True)
tag = "" if a.src == "all" else "_" + a.src
pd.DataFrame(rows).to_csv(out / f"w2_early_signal{tag}.csv", index=False)
C = pd.DataFrame(cov); C.to_csv(out / f"w2_coverage{tag}.csv", index=False)
print(C.pivot(index="h", columns="period", values="covered").round(3).to_string())
print(C.assign(gap=C.mean_log_cov - C.mean_log_unc).pivot(index="h", columns="period", values="gap").round(3).to_string())
