"""Which network is the final share count made of, and is "early Facebook ranks articles best" only composition?

The dataset's `shares` is one number. Internet Archive captures of the article pages (data/wayback/, src ==
"article") show the six per-network counters long after publication. For each article (repaired URL) we take the
article-page capture nearest to 2015-01-08, the day the dataset's shares were collected, and keep it only if its
total agrees with `shares` (exactly below 1,000; within 50 above, where the dataset rounds to hundreds).

Part 1  composition of the final count: each network's part of all shares and of the median article, for all
        articles and for the top 5% by shares; the spread (sd of log1p) of each network's final count. Covered
        articles are slightly more popular than the rest, so the composition is also given reweighted to the whole
        dataset (inverse coverage within 20 quantile bins of shares).
Part 2  cross-network prediction, past-only (monthly rolling origin as in v8; HistGB on the age at the look and ONE
        early count, fitted on earlier months): the early state is the last homepage sighting within h hours (3, 6).
        Target "later growth outside Facebook" = final count without Facebook minus early count without Facebook. If
        early Facebook's advantage were only that Facebook dominates the final count, it would not rank this target
        better than early Twitter does. Scored on 2013-07..2014-12: AUC for exceeding the target's median, Spearman
        correlation with the log target, and the paired Facebook-minus-Twitter difference (bootstrap over days).

Output: results/tables/v9_final_composition.csv, v9_cross_network.csv
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402

NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]
FIRST, LAST, B = "2013-07", "2014-12", 1000
OUT = ROOT / "results" / "tables"

df = load().reset_index(drop=True)
n = len(df)
s = df.shares.to_numpy().astype(float)
date = pd.to_datetime(df.publish_date)
month = date.dt.to_period("M").astype(str).to_numpy()
day = date.dt.strftime("%Y-%m-%d").to_numpy()
t95 = np.quantile(s, .95)
aid = df.url.fillna("").str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2}/[^/]+)")[0].str.lower()
row_of = pd.Series(np.arange(n), index=aid.to_numpy())
row_of = row_of[row_of.index.notna() & ~row_of.index.duplicated()]

W = ROOT / "data" / "wayback"
O = pd.read_csv(W / "wb_obs.csv.gz", dtype={"ts": str})
M = pd.read_csv(W / "wb_meta.csv.gz")
M["post_date"] = pd.to_datetime(M.post_date, utc=True, errors="coerce").dt.tz_localize(None)
O["ts"] = pd.to_datetime(O.ts, format="%Y%m%d%H%M%S")
O["row"] = O.art_id.map(row_of)
O = O.dropna(subset=["row"]).astype({"row": int})

# ---- final per-network counts from the article pages ----------------------------------------------------
A = O[(O.src == "article") & O.total.notna()].copy()
A["absgap"] = (A.ts - pd.Timestamp("2015-01-08")).abs()
F = A.sort_values("absgap").groupby("row").head(1).set_index("row").sort_index()
F[NETS] = F[NETS].fillna(0)
F["shares"] = s[F.index]
agree = np.where(F.shares < 1000, F.total == F.shares, (F.total - F.shares).abs() <= 50)
G = F[agree].copy()
cov = np.zeros(n, bool); cov[G.index] = True
print(f"articles {n:,}; with an article-page capture {len(F):,}; nearest capture agrees with shares {len(G):,} "
      f"({len(G) / n:.1%} of the dataset); reach 1,400: covered {np.mean(s[cov] >= 1400):.1%}, "
      f"not covered {np.mean(s[~cov] >= 1400):.1%}")

# inverse-coverage weights within 20 quantile bins of shares
bins = pd.qcut(pd.Series(s).rank(method="first"), 20, labels=False).to_numpy()
rate = pd.Series(cov).groupby(bins).mean()
G["w"] = 1 / rate.reindex(bins[G.index]).to_numpy()

rows = []
for name, m in [("all", np.ones(len(G), bool)), ("below 1,400", (G.shares < 1400).to_numpy()),
                ("top 5%", (G.shares >= t95).to_numpy())]:
    g = G[m]
    for k in NETS:
        rows.append(dict(group=name, network=k, n=len(g),
                         part_of_all_shares=g[k].sum() / g.total.sum(),
                         part_of_all_shares_reweighted=(g[k] * g.w).sum() / (g.total * g.w).sum(),
                         part_of_median_article=(g[k] / g.total.clip(lower=1)).median(),
                         median_count=g[k].median(), sd_log1p=np.log1p(g[k]).std()))
T1 = pd.DataFrame(rows); T1.to_csv(OUT / "v9_final_composition.csv", index=False)
print(T1.round(3).to_string(index=False))

# ---- early homepage look against the final components ----------------------------------------------------
H = O[O.src == "home"].merge(M[["art_id", "post_date"]], on="art_id", how="left").drop_duplicates(["art_id", "ts", "col"])
H["age_h"] = (H.ts - H.post_date).dt.total_seconds() / 3600
H = H[(H.age_h > 0) & H.total.notna()].sort_values(["row", "ts"])


def ci(fn, d, name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    days = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in days}
    v = [fn(np.concatenate([idx[x] for x in rng.choice(days, len(days))])) for _ in range(B)]
    return np.quantile(v, [.025, .975])


cross = []
for h in (3.0, 6.0):
    last = H[H.age_h <= h].groupby("row").tail(1).set_index("row")
    both = last.index.intersection(G.index)
    e = last.loc[both]; g = G.loc[both]
    e_net = {k: e[k].fillna(0).to_numpy() for k in NETS}; e_tot = e.total.to_numpy()
    mth, dd = month[both], day[both]
    use = (mth >= FIRST) & (mth <= LAST)
    targets = {
        "final total": g.total.to_numpy(),
        "final facebook": g.facebook.to_numpy(),
        "later growth outside facebook": np.clip((g.total - g.facebook).to_numpy() - (e_tot - e_net["facebook"]), 0, None),
        "later growth outside twitter": np.clip((g.total - g.twitter).to_numpy() - (e_tot - e_net["twitter"]), 0, None),
    }
    feats = {"early facebook": np.c_[e.age_h, np.log1p(e_net["facebook"])],
             "early twitter": np.c_[e.age_h, np.log1p(e_net["twitter"])],
             "early total": np.c_[e.age_h, np.log1p(e_tot)]}
    print(f"\nlook within {h:g} h: {len(both):,} articles with an early look and a final decomposition, "
          f"{use.sum():,} scored; early count holds {e_tot[use].sum() / g.total.to_numpy()[use].sum():.1%} of the final "
          f"count; early twitter {e_net['twitter'][use].sum() / e_tot[use].sum():.1%}, "
          f"facebook {e_net['facebook'][use].sum() / e_tot[use].sum():.1%}")
    for tn, tv in targets.items():
        yt = np.log1p(tv.astype(float)); lab = (yt >= np.median(yt[use]))[use]; du = dd[use]
        pred = {}
        for fn, Xf in feats.items():
            p = np.full(len(both), np.nan)
            for m in np.unique(mth[use]):
                tr, te = mth < m, mth == m
                p[te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(Xf[tr], yt[tr]).predict(Xf[te])
            pred[fn] = p[use]
            lo, hi = ci(lambda t, q=pred[fn]: roc_auc_score(lab[t], q[t]), du, f"{h}|{tn}|{fn}")
            cross.append(dict(h=h, target=tn, signal=fn, n=int(use.sum()), auc_above_median=roc_auc_score(lab, pred[fn]),
                              lo=lo, hi=hi, spearman=spearmanr(pred[fn], yt[use])[0]))
        pf, pt = pred["early facebook"], pred["early twitter"]
        lo, hi = ci(lambda t: roc_auc_score(lab[t], pf[t]) - roc_auc_score(lab[t], pt[t]), du, f"{h}|{tn}|diff")
        cross.append(dict(h=h, target=tn, signal="facebook minus twitter", n=int(use.sum()),
                          auc_above_median=roc_auc_score(lab, pf) - roc_auc_score(lab, pt), lo=lo, hi=hi,
                          spearman=spearmanr(pf, yt[use])[0] - spearmanr(pt, yt[use])[0]))
T2 = pd.DataFrame(cross); T2.to_csv(OUT / "v9_cross_network.csv", index=False)
print(T2.round(3).to_string(index=False))
