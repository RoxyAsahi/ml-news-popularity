"""Direction A, article pages: what the FINAL share count is made of, and whether "Facebook carries the early signal"
is only a composition effect.

The archive's captures of the article pages give the six per-network counts long after publication. For each dataset
article take the article-page capture nearest to 2015-01-08 (the day the dataset's `shares` were collected) and keep
it only if its total agrees with `shares` (exactly below 1,000; within 50 above, where the dataset rounds to hundreds).

Part 1  composition of the final count: share-weighted and per-article, for all / below 1,400 / at least 1,400 /
        top 5% articles; spread of each network's final count; how well each final component alone ranks the label.
Part 2  between the early homepage look (last sighting at age <= h) and the final count: which network the later
        growth comes from.
Part 3  cross-network prediction, past-only (monthly rolling origin, fit on months < m, score month m, HGB on age + the
        early count(s)): does the early Facebook count predict the final count OUTSIDE Facebook, and does the early
        Twitter count predict the final count outside Twitter? Targets are log final components; reported as AUC for
        exceeding the component's median and as Spearman correlation, with day-level bootstrap intervals.

Outputs: results/tables/w9_final_composition.csv, w9_growth_by_network.csv, w9_cross_network.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HGB_PARAMS, NETS, ROOT, early_state, load_fixed, load_obs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2013-07")
ap.add_argument("--last", default="2014-12")
ap.add_argument("--horizons", default="3,6")
ap.add_argument("--boot", type=int, default=500)
a = ap.parse_args()
OUT = ROOT / "results" / "tables"
rng = np.random.default_rng(0)

df = load_fixed()
n = len(df)
s = df.shares.to_numpy().astype(float)
month = df.date.dt.to_period("M").astype(str).to_numpy()
day = df.date.dt.strftime("%Y-%m-%d").to_numpy()
t95 = np.quantile(s, .95)

# ---- final per-network counts from the article pages -------------------------------------------------------------
R = pd.read_parquet(ROOT / "data" / "interim" / "wb_obs.parquet")
row_of = pd.Series(np.arange(n), index=df.art_id.to_numpy())
row_of = row_of[row_of.index.notna() & ~row_of.index.duplicated()]
A = R[(R.src == "article") & R.total.notna()].copy()
A["row"] = A.art_id.map(row_of)
A = A.dropna(subset=["row"]).astype({"row": int})
A["gap_d"] = (A.ts - pd.Timestamp("2015-01-08")).dt.total_seconds() / 86400
A["absgap"] = A.gap_d.abs()
F = A.sort_values("absgap").groupby("row").head(1).set_index("row").sort_index()
F["shares"] = s[F.index]
F[NETS] = F[NETS].fillna(0)
agree = np.where(F.shares < 1000, F.total == F.shares, (F.total - F.shares).abs() <= 50)
print(f"dataset articles {n:,}; with an article-page capture {len(F):,}; nearest capture agrees with `shares` "
      f"{agree.sum():,} ({agree.mean():.1%} of those, {agree.sum() / n:.1%} of the dataset)")
print("  agreement by distance of the capture from 2015-01-08 (days):")
band = pd.cut(F.gap_d, [-800, -180, -30, -3, 3, 30, 400])
print((pd.Series(agree, index=F.index).groupby(band, observed=True).agg(["size", "mean"]).round(3)).to_string())
G = F[agree].copy()
cov = np.zeros(n, bool); cov[G.index] = True
print(f"  covered articles: {np.mean(s[cov] >= 1400):.1%} reach 1,400; not covered: {np.mean(s[~cov] >= 1400):.1%}")

# ---- Part 1: composition ------------------------------------------------------------------------------------------
print("\nPart 1  composition of the final count")
rows = []
for name, m in [("all", np.ones(len(G), bool)), ("below 1,400", (G.shares < 1400).to_numpy()),
                ("at least 1,400", (G.shares >= 1400).to_numpy()), ("top 5%", (G.shares >= t95).to_numpy())]:
    g = G[m]
    r = dict(group=name, n=len(g), median_total=g.total.median())
    for k in NETS:
        r["w_" + k] = g[k].sum() / g.total.sum()                       # share of all shares in the group
        r["med_" + k] = (g[k] / g.total.clip(lower=1)).median()         # median article's share
    rows.append(r)
T1 = pd.DataFrame(rows)
print(T1.round(3).to_string(index=False))
spread = pd.DataFrame({"median": G[NETS].median(), "q90": G[NETS].quantile(.9), "mean": G[NETS].mean(),
                       "sd_log1p": np.log1p(G[NETS]).std(),
                       "auc_alone_1400": [roc_auc_score(G.shares >= 1400, G[k]) for k in NETS],
                       "auc_rest_1400": [roc_auc_score(G.shares >= 1400, G.total - G[k]) for k in NETS]})
print("\n  each network's final count: spread, and AUC for 'total >= 1,400' of that network alone / of the rest")
print(spread.round(3).to_string())
T1.to_csv(OUT / "w9_final_composition.csv", index=False)
spread.to_csv(OUT / "w9_final_spread.csv")

# ---- Parts 2 and 3: early look against the final components -------------------------------------------------------
O = load_obs(df, lists_only=True)


def reg():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


def boot_ci(fn, d):
    days = np.unique(d)
    idx = {x: np.flatnonzero(d == x) for x in days}
    v = []
    for _ in range(a.boot):
        take = np.concatenate([idx[x] for x in rng.choice(days, len(days))])
        try:
            v.append(fn(take))
        except ValueError:
            pass
    return np.quantile(v, [.025, .975])


growth, cross = [], []
for h in [float(x) for x in a.horizons.split(",")]:
    E = early_state(O, h)
    both = E.index.intersection(G.index)
    E = E.loc[both]; g = G.loc[both]
    e_cnt = pd.DataFrame({k: np.expm1(E["l_" + k]) for k in NETS}, index=both)
    e_tot = np.expm1(E.l_total)
    print(f"\nPart 2  look within {h:g} h: {len(both):,} articles with both an early look and a final decomposition "
          f"(median age at the look {E.age.median():.1f} h)")
    d_tot = (g.total - e_tot).clip(lower=0)
    for name, m in [("all", np.ones(len(g), bool)), ("at least 1,400", (g.shares >= 1400).to_numpy()),
                    ("top 5%", (g.shares >= t95).to_numpy())]:
        r = dict(h=h, group=name, n=int(m.sum()), early_share_of_final=e_tot[m].sum() / g.total[m].sum())
        for k in NETS:
            r["early_" + k] = e_cnt[k][m].sum() / e_tot[m].sum()
            r["final_" + k] = g[k][m].sum() / g.total[m].sum()
            r["growth_" + k] = (g[k] - e_cnt[k]).clip(lower=0)[m].sum() / sum((g[j] - e_cnt[j]).clip(lower=0)[m].sum() for j in NETS)
        growth.append(r)
    P = pd.DataFrame(growth); P = P[P.h == h]
    for name in P.group:
        r = P[P.group == name].iloc[0]
        print(f"  {name:15s} n={r.n:6d}  seen by the look: {r.early_share_of_final:.1%} of the final count")
        for what in ["early", "final", "growth"]:
            print(f"     {what:7s}" + "  ".join(f"{k}:{r[what + '_' + k]:.3f}" for k in NETS))

    # Part 3: past-only models from one early count to each final component
    rr = both.to_numpy()
    mth = month[rr]; dd = day[rr]
    use = (mth >= a.first) & (mth <= a.last)
    targets = {"total": g.total, "facebook": g.facebook, "twitter": g.twitter,
               "outside facebook": g.total - g.facebook, "outside twitter": g.total - g.twitter,
               "outside facebook and twitter": g.total - g.facebook - g.twitter,
               "later growth outside facebook": ((g.total - g.facebook) - (e_tot - e_cnt.facebook)).clip(lower=0),
               "later growth outside twitter": ((g.total - g.twitter) - (e_tot - e_cnt.twitter)).clip(lower=0)}
    feats = {"early facebook": ["age", "l_facebook"], "early twitter": ["age", "l_twitter"],
             "early total": ["age", "l_total"], "early all six": ["age"] + ["l_" + k for k in NETS]}
    print(f"\nPart 3  look within {h:g} h, months {a.first}..{a.last}: {use.sum():,} articles")
    print(f"  {'final target':32s}" + "".join(f"{f:>26s}" for f in feats))
    for tn, tv in targets.items():
        yt = np.log1p(tv.to_numpy().astype(float))
        pred = {}
        for fn, cols in feats.items():
            Xf = E[cols].to_numpy()
            p = np.full(len(rr), np.nan)
            for m in np.unique(mth[use]):
                tr = mth < m; te = mth == m
                p[te] = reg().fit(Xf[tr], yt[tr]).predict(Xf[te])
            pred[fn] = p
        hi = yt >= np.median(yt[use])
        line = f"  {tn:32s}"
        for fn in feats:
            p = pred[fn][use]; lab = hi[use]; du = dd[use]
            auc = roc_auc_score(lab, p)
            lo, up = boot_ci(lambda t: roc_auc_score(lab[t], p[t]), du)
            rho = spearmanr(p, yt[use])[0]
            cross.append(dict(h=h, target=tn, features=fn, n=int(use.sum()), auc_above_median=auc, lo=lo, hi=up,
                              spearman=rho))
            line += f"   {auc:.3f} [{lo:.3f},{up:.3f}] r={rho:.2f}"
        # the composition question as one paired difference: early facebook minus early twitter on this target
        pf, pt = pred["early facebook"][use], pred["early twitter"][use]; lab = hi[use]; du = dd[use]
        dlo, dup = boot_ci(lambda t: roc_auc_score(lab[t], pf[t]) - roc_auc_score(lab[t], pt[t]), du)
        cross.append(dict(h=h, target=tn, features="facebook minus twitter", n=int(use.sum()),
                          auc_above_median=roc_auc_score(lab, pf) - roc_auc_score(lab, pt), lo=dlo, hi=dup,
                          spearman=np.nan))
        print(line + f"   fb-tw {cross[-1]['auc_above_median']:+.3f} [{dlo:+.3f},{dup:+.3f}]", flush=True)

pd.DataFrame(growth).to_csv(OUT / "w9_growth_by_network.csv", index=False)
pd.DataFrame(cross).to_csv(OUT / "w9_cross_network.csv", index=False)
print("\nwrote results/tables/w9_*.csv")
