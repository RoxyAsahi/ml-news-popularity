"""Direction A, deep dive 1: an outside audit of Mashable's own "The Next Big Thing" (Velocity) signal,
plus what each social network and the newsroom metadata add.

In 2013-14 the Mashable homepage sorted recent articles into three columns: The New Stuff (state "new"),
The Next Big Thing ("rising", chosen by the proprietary Velocity model, with a 0-1000 velocity series) and
What's Hot ("hot"). The archive captures give each article's state and exact share counts at the capture time;
the dataset gives the final count (collected 2015-01-08).

Part 1  state table: for each age band and state, how often does the article end >= 1400 / in the top 5%?
Part 2  ablation, past-only monthly rolling origin, at a look h hours after publication (last sighting <= h):
          counts            age + share counts per network
          counts+state      + Mashable's state (column, rank, ever rising / ever hot)
          counts+state+vel  + velocity values
          mashable_only     age + state + velocity, no share counts
          total_only        age + total count only (no per-network split)
          prepub            the 58 dataset features
        differences against `counts` with a bootstrap over publication days.
Part 3  inside the rising column: does the velocity value predict the final count beyond count and age?
Part 4  networks: mix of the early count by age; later growth by early Facebook share.
Part 5  newsroom metadata known at publication (author with a past-only encoding, hour of day, text length)
        added to the 58 features, on the articles for which the archive gives the metadata.

Outputs: results/tables/w3_state_table.csv, w3_ablation.csv, w3_velocity_inside.csv, w3_networks.csv, w3_metadata.csv
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
from common import (COUNT_COLS, HGB_PARAMS, NETS, ROOT, STATE_COLS, VEL_COLS, early_state, features, load_fixed,  # noqa: E402
                    load_obs)

ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2013-04")
ap.add_argument("--horizons", default="1,2,3,6,12,24")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--parts", default="1,2,3,4,5", help="part 2 (the ablation) is the slow one")
a = ap.parse_args()
PARTS = a.parts.split(",")
OUT = ROOT / "results" / "tables"

df = load_fixed()
n = len(df)
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float)
y = np.log1p(s)
month = df.date.dt.to_period("M").astype(str).to_numpy()
dn = (df.date - df.date.min()).dt.days.to_numpy()
t95 = np.quantile(s, .95)
O = load_obs(df, lists_only=True)
O["final"] = s[O.row.to_numpy()]
last_month = O.ts.max().strftime("%Y-%m")
print(f"sightings {len(O):,} on {O.row.nunique():,} dataset articles, captures up to {O.ts.max()}", flush=True)
rng = np.random.default_rng(0)


def reg():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


def fill_dead(A):
    A = A.copy(); A[:, np.isnan(A).all(0)] = 0.0
    return A


# ------------------------------------------------------------------ Part 1: state table
bands = [0, 1, 2, 4, 8, 16, 32, 96]
O["band"] = pd.cut(O.age_h, bands, right=False)
one = O.dropna(subset=["band"]).groupby(["band", "col", "row"], observed=True).head(1)  # one sighting per article
T = one.groupby(["band", "col"], observed=True).agg(
    n=("row", "size"), cur_median=("total", "median"), final_median=("final", "median"),
    p1400=("final", lambda v: (v >= 1400).mean()), p_top5=("final", lambda v: (v >= t95).mean()),
    already1400=("total", lambda v: (v >= 1400).mean())).reset_index()
T["band"] = T.band.astype(str)
T.to_csv(OUT / "w3_state_table.csv", index=False)
print("\nPart 1: final outcome by age band and Mashable state (base rates: >=1400 %.3f, top5 %.3f)" %
      ((s >= 1400).mean(), (s >= t95).mean()))
print(T.round(3).to_string(index=False), flush=True)

# ------------------------------------------------------------------ Part 2: ablation
if "2" in PARTS:
    SETS = {
        "counts": COUNT_COLS,
        "counts+state": COUNT_COLS + STATE_COLS,
        "counts+state+vel": COUNT_COLS + STATE_COLS + VEL_COLS,
        "mashable_only": ["age"] + STATE_COLS + VEL_COLS,
        "total_only": ["age", "l_total", "l_rate0"],
    }
    rows = []
    for h in [float(x) for x in a.horizons.split(",")]:
        E = early_state(O, h)
        idx = E.index.to_numpy(); mi = month[idx]; yi = y[idx]; si = s[idx]; di = dn[idx]
        sc = {k: np.full(len(idx), np.nan) for k in list(SETS) + ["prepub", "prepub+counts", "prepub+all"]}
        mats = {k: fill_dead(E[c].to_numpy(dtype=float)) for k, c in SETS.items()}
        mats["prepub"] = X[idx]
        mats["prepub+counts"] = np.c_[X[idx], mats["counts"]]
        mats["prepub+all"] = np.c_[X[idx], mats["counts+state+vel"]]
        for m in sorted(set(mi)):
            tr, te = mi < m, mi == m
            if m < a.first or tr.sum() < 300 or te.sum() == 0:
                continue
            for k, A in mats.items():
                sc[k][te] = reg().fit(A[tr], yi[tr]).predict(A[te])
        ok = ~np.isnan(sc["counts"])
        days = np.unique(di[ok]); gi = [np.flatnonzero(di[ok] == d) for d in days]
        boot = rng.integers(0, len(days), size=(a.boot, len(days)))

        def met(z, sub):
            return np.array([roc_auc_score(si[ok][sub] >= 1400, z[sub]),
                             roc_auc_score(si[ok][sub] >= t95, z[sub]) if (si[ok][sub] >= t95).any() else np.nan,
                             spearmanr(z[sub], yi[ok][sub])[0], r2_score(yi[ok][sub], z[sub])])

        full = np.arange(ok.sum())
        base = sc["counts"][ok]
        for k in sc:
            z = sc[k][ok]
            v = met(z, full)
            D = np.array([met(z, sub) - met(base, sub) for sub in (np.concatenate([gi[j] for j in b]) for b in boot)])
            lo, hi = np.nanquantile(D, [.025, .975], axis=0)
            rows.append(dict(h=h, features=k, n=int(ok.sum()), auc1400=v[0], auc_top5=v[1], spearman=v[2], r2=v[3],
                             d_auc1400=v[0] - met(base, full)[0], d_auc1400_lo=lo[0], d_auc1400_hi=hi[0],
                             d_r2=v[3] - met(base, full)[3], d_r2_lo=lo[3], d_r2_hi=hi[3]))
        print(pd.DataFrame(rows[-len(sc):]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "w3_ablation.csv", index=False)

# ------------------------------------------------------------------ Part 3: velocity inside the rising column
R = O[(O.col == "rising") & O.vel_last.notna()].groupby("row").head(1).copy()  # first rising sighting per article
R["ly"] = np.log1p(R.final); R["lc"] = np.log1p(R.total); R["la"] = np.log(R.age_h.clip(lower=.05))
ctrl = np.c_[R.lc, R.la, R.lc * R.la, R.lc ** 2, R.la ** 2]
res_y = R.ly - LinearRegression().fit(ctrl, R.ly).predict(ctrl)
vrows = []
for k in ["vel_last", "vel_max", "vel_slope", "vel_min", "rank"]:
    v = R[k].to_numpy(dtype=float)
    res_v = v - LinearRegression().fit(ctrl, v).predict(ctrl)
    vrows.append(dict(signal=k, n=len(R), raw_spearman=spearmanr(v, R.ly)[0], partial_spearman=spearmanr(res_v, res_y)[0],
                      spearman_with_growth=spearmanr(v, R.ly - R.lc)[0]))
for k, v in [("count_now", R.lc), ("age", R.la)]:
    vrows.append(dict(signal=k, n=len(R), raw_spearman=spearmanr(v, R.ly)[0], partial_spearman=np.nan,
                      spearman_with_growth=spearmanr(v, R.ly - R.lc)[0]))
V = pd.DataFrame(vrows); V.to_csv(OUT / "w3_velocity_inside.csv", index=False)
print("\nPart 3: inside The Next Big Thing (first rising sighting per article)")
print(V.round(3).to_string(index=False), flush=True)

# ------------------------------------------------------------------ Part 4: networks
one = O.dropna(subset=["band"]).groupby(["band", "row"], observed=True).head(1).copy()
tot = one[NETS].sum(axis=1).clip(lower=1)
for k in NETS:
    one["sh_" + k] = one[k] / tot
N1 = one.groupby("band", observed=True)[["sh_" + k for k in NETS]].mean()
N1["n"] = one.groupby("band", observed=True).size()
N1["count_vs_final"] = one.assign(r=one.total / one.final.clip(lower=1)).groupby("band", observed=True).r.median()
mid = one[(one.age_h >= 2) & (one.age_h < 12) & (one.total >= 50)].copy()
mid["fbq"] = pd.qcut(mid.sh_facebook, 5, labels=False, duplicates="drop")
mid["growth"] = np.log1p(mid.final) - np.log1p(mid.total)
N2 = mid.groupby("fbq").agg(n=("row", "size"), fb_share=("sh_facebook", "mean"), cur_median=("total", "median"),
                            growth_mean=("growth", "mean"), final_median=("final", "median"),
                            p_top5=("final", lambda v: (v >= t95).mean()))
ctrl = np.c_[np.log1p(mid.total), np.log(mid.age_h), np.log1p(mid.total) * np.log(mid.age_h)]
rg = mid.growth - LinearRegression().fit(ctrl, mid.growth).predict(ctrl)
rf = mid.sh_facebook - LinearRegression().fit(ctrl, mid.sh_facebook).predict(ctrl)
print("\nPart 4: network mix of the count by age band")
print(N1.round(3).to_string())
print("later growth (log final - log now) by early Facebook share, sightings at 2-12 h:")
print(N2.round(3).to_string())
print("partial Spearman of Facebook share with later growth, given count and age: %.3f" % spearmanr(rf, rg)[0], flush=True)
pd.concat([N1.reset_index().assign(table="mix_by_age", band=lambda d: d.band.astype(str)),
           N2.reset_index().assign(table="growth_by_fb_share")]).to_csv(OUT / "w3_networks.csv", index=False)

# ------------------------------------------------------------------ Part 5: newsroom metadata
M = pd.read_parquet(ROOT / "data" / "interim" / "wb_meta.parquet").drop_duplicates("art_id").set_index("art_id")
J = df[["art_id"]].join(M[["post_date", "author", "n_chars"]], on="art_id")
has = J.post_date.notna().to_numpy() & (month <= last_month)
pt = pd.to_datetime(J.post_date)
hour = (pt.dt.hour + pt.dt.minute / 60).to_numpy()
how = (pt.dt.day_of_week * 24 + pt.dt.hour).to_numpy(dtype=float)
auth = J.author.fillna("").str.strip().str.replace(r"\s+", " ", regex=True).to_numpy()
# gap to the previous article on the site (minutes), from exact publication times of the sighted articles
order = np.argsort(pt.to_numpy()); gap = np.full(n, np.nan)
tt = pt.to_numpy()[order].astype("datetime64[s]").astype(float)
gap[order[1:]] = np.diff(tt) / 60.0
gap[~has] = np.nan


def author_enc(tr, te):
    """Shrunk mean of log shares and article count of the author over the training rows (earlier months)."""
    g = pd.DataFrame({"a": auth[tr], "y": y[tr]}).groupby("a").y.agg(["sum", "size"])
    mu = y[tr].mean()
    enc = ((g["sum"] + 10 * mu) / (g["size"] + 10)).to_dict(); cnt = g["size"].to_dict()
    f = lambda ix: np.c_[[enc.get(x, mu) for x in auth[ix]], [cnt.get(x, 0) for x in auth[ix]]]
    # training rows: leave-one-out version to avoid target leakage
    yl = y[tr]; al = auth[tr]
    su = np.array([g["sum"].get(x, 0) for x in al]); sz = np.array([g["size"].get(x, 0) for x in al])
    loo = np.c_[(su - yl + 10 * mu) / (sz - 1 + 10), sz - 1]
    return loo, f(te)


idx = np.flatnonzero(has); mi = month[idx]
names = ["prepub58", "+hour", "+author", "+hour+author+length"]
sc = {k: np.full(len(idx), np.nan) for k in names}
for m in sorted(set(mi)):
    trm, tem = mi < m, mi == m
    if m < a.first or trm.sum() < 300 or tem.sum() == 0:
        continue
    tr, te = idx[trm], idx[tem]
    a_tr, a_te = author_enc(tr, te)
    tm = lambda ix: np.c_[hour[ix], how[ix]]
    ln = lambda ix: np.c_[np.log1p(np.nan_to_num(J.n_chars.to_numpy(dtype=float)[ix], nan=0.0)), gap[ix]]
    packs = {"prepub58": (X[tr], X[te]), "+hour": (np.c_[X[tr], tm(tr)], np.c_[X[te], tm(te)]),
             "+author": (np.c_[X[tr], a_tr], np.c_[X[te], a_te]),
             "+hour+author+length": (np.c_[X[tr], tm(tr), a_tr, ln(tr)], np.c_[X[te], tm(te), a_te, ln(te)])}
    for k, (A, B) in packs.items():
        sc[k][tem] = reg().fit(A, y[tr]).predict(B)
ok = ~np.isnan(sc["prepub58"]); si, yi, di = s[idx][ok], y[idx][ok], dn[idx][ok]
days = np.unique(di); gi = [np.flatnonzero(di == d) for d in days]
boot = rng.integers(0, len(days), size=(a.boot, len(days)))


def met5(z, sub):
    return np.array([roc_auc_score(si[sub] >= 1400, z[sub]), roc_auc_score(si[sub] >= t95, z[sub]),
                     spearmanr(z[sub], yi[sub])[0], r2_score(yi[sub], z[sub])])


full = np.arange(ok.sum()); mrows = []
for k in names:
    z = sc[k][ok]; v = met5(z, full); b0 = sc["prepub58"][ok]
    D = np.array([met5(z, sub) - met5(b0, sub) for sub in (np.concatenate([gi[j] for j in b]) for b in boot)])
    lo, hi = np.quantile(D, [.025, .975], axis=0)
    mrows.append(dict(features=k, n=int(ok.sum()), auc1400=v[0], auc_top5=v[1], spearman=v[2], r2=v[3],
                      d_auc1400=v[0] - met5(b0, full)[0], d_auc1400_lo=lo[0], d_auc1400_hi=hi[0],
                      d_spearman=v[2] - met5(b0, full)[2], d_spearman_lo=lo[2], d_spearman_hi=hi[2]))
MM = pd.DataFrame(mrows); MM.to_csv(OUT / "w3_metadata.csv", index=False)
print("\nPart 5: metadata known at publication, on %d archive-covered articles (%d authors)" % (ok.sum(), len(set(auth[idx]))))
print(MM.round(4).to_string(index=False))
