"""How much do the first hours after publication add to what can be called before it? (repaired data)

Internet Archive captures of Mashable's homepage (data/wayback/, src == "home") show each listed article with its
share count per network at capture time. For a look horizon h (1, 3, 6, 12, 24 hours) the early state of an article
is its LAST homepage sighting at an age of at most h hours: age, log counts (total and six networks), Facebook's
part of the total, log shares per hour, the homepage column (new / rising / hot), position in the column and the
number of sightings so far. Articles not sighted by then have no early state (missing values).

Protocol: the monthly rolling origin of v1_past_only_called.py. The pre-publication score is `ens` (mean of HistGB
on the 58 features and the headline-embedding ridge), produced past-only for every month from 2013-03 (months from
2013-07 are read from v1's scores, the four earlier months are fitted here the same way). For each month m from
2013-07 and each h, HistGB (same settings) is fitted on the articles of 2013-03 .. m-1 with the 58 features, the
pre-publication score and the early state at h, and scores month m. Rows and metrics are those of the main table
(articles of 2013-07 .. 2014-12 with a headline; day bootstrap).

Reported per h: the metrics on ALL articles (unseen ones included); the same with captures thinned to at most 4 per
day; the share of articles sighted by h and the median age of that sighting; the share already past 1,400 shares at
the look; and, among sighted articles not yet past 1,400, the AUC of the model, of the pre-publication score and of
the current count alone (the constant-ratio extrapolation of Szabo & Huberman 2010 ranks articles by it).
Also, among articles sighted by 3 h: AUC of the Facebook count, the Twitter count and the total alone.
Output: results/tables/v8_first_hours.csv, v8_first_hours_diffs.csv, v8_first_hours_coverage.csv,
v8_networks.csv, data/interim/v8_scores.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.dayeval import evaluate  # noqa: E402
from newspop.titles import aligned_titles, bge_embeddings  # noqa: E402

HORIZONS = [1, 3, 6, 12, 24]
NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]
FIRST, START = "2013-07", "2013-03"
T = ROOT / "results" / "tables"

df = load().reset_index(drop=True); n = len(df)
y = np.log1p(df.shares.to_numpy()); shares = df.shares.to_numpy().astype(float)
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number").to_numpy()
day = pd.to_datetime(df.publish_date).dt.normalize(); month = day.dt.to_period("M").astype(str).to_numpy()
months = sorted(set(month))


def hgb():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


# --- pre-publication score for every month from START (past-only) --------------------------------------
V1 = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv")
pre = np.full(n, np.nan); pre[V1.row] = V1.ens
E = bge_embeddings(aligned_titles(df).fillna(""))
for m in [m for m in months if START <= m < FIRST]:
    tr, te = month < m, month == m
    sc = StandardScaler().fit(E[tr])
    pre[te] = (hgb().fit(X[tr], y[tr]).predict(X[te]) + Ridge(alpha=1000).fit(sc.transform(E[tr]), y[tr]).predict(sc.transform(E[te]))) / 2
del E

# --- homepage sightings -----------------------------------------------------------------------------
W = ROOT / "data" / "wayback"
O = pd.read_csv(W / "wb_obs.csv.gz", dtype={"ts": str}); M = pd.read_csv(W / "wb_meta.csv.gz")
M["post_date"] = pd.to_datetime(M.post_date, utc=True, errors="coerce").dt.tz_localize(None)
O = O[O.src == "home"].copy(); O["ts"] = pd.to_datetime(O.ts, format="%Y%m%d%H%M%S")
O = O.merge(M[["art_id", "post_date"]], on="art_id", how="left").drop_duplicates(["art_id", "ts", "col"])
O["age_h"] = (O.ts - O.post_date).dt.total_seconds() / 3600
O = O[(O.age_h > 0) & O.total.notna()]
aid = df.url.fillna("").str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2}/[^/]+)")[0].str.lower()
row_of = pd.Series(np.arange(n), index=aid.to_numpy()); row_of = row_of[row_of.index.notna() & ~row_of.index.duplicated()]
O["row"] = O.art_id.map(row_of); O = O.dropna(subset=["row"]).astype({"row": int}).sort_values(["row", "ts"])
print(f"homepage captures {O.ts.nunique():,}, sightings {len(O):,}, articles {O.row.nunique():,}", flush=True)


def thin(O, k=4):
    """Keep at most k captures per calendar day, evenly spaced."""
    cap = pd.Series(np.sort(O.ts.unique())); keep = []
    for _, g in cap.groupby(cap.dt.normalize()):
        keep += list(g.iloc[np.unique(np.round(np.linspace(0, len(g) - 1, min(k, len(g)))).astype(int))])
    return O[O.ts.isin(keep)]


def early_state(O, h):
    o = O[O.age_h <= h]; g = o.groupby("row"); last = g.tail(1).set_index("row")
    S = pd.DataFrame(index=last.index)
    S["age"] = last.age_h; S["total"] = last.total; S["l_total"] = np.log1p(last.total)
    for k in NETS:
        S["l_" + k] = np.log1p(last[k].fillna(0))
    S["fb_part"] = last.facebook.fillna(0) / (last.total + 1)
    S["l_rate"] = S.l_total - np.log(S.age.clip(lower=0.05))
    for c in ["new", "rising", "hot"]:
        S["col_" + c] = (last.col == c).astype(float)
    S["rank"] = last["rank"]; S["n_sight"] = g.size()
    return S.reindex(np.arange(n))


def rolling(Z):
    s = np.full(n, np.nan)
    for m in [m for m in months if m >= FIRST]:
        tr, te = (month >= START) & (month < m), month == m
        s[te] = hgb().fit(Z[tr], y[tr]).predict(Z[te])
    return s


S = V1[["row", "y", "pop", "day", "chan", "has_title"]].copy(); S["pre"] = pre[S.row]
info, sub, states = [], [], {}
for name, Oh in [("", O), ("_thin4", thin(O))]:
    for h in HORIZONS:
        st = early_state(Oh, h); states[(name, h)] = st
        S[f"h{h}{name}"] = rolling(np.c_[X, pre, st.drop(columns="total").to_numpy()])[S.row]
        print("scored", h, name or "all captures", flush=True)
S.to_csv(ROOT / "data" / "interim" / "v8_scores.csv", index=False)
Sv = S[S.has_title].reset_index(drop=True)
MODELS = ["pre"] + [f"h{h}{nm}" for nm in ["", "_thin4"] for h in HORIZONS]
R, D = evaluate(Sv, MODELS, base=["pre"])
rng = np.random.default_rng(0)
for h in HORIZONS:
    st = states[("", h)].iloc[Sv.row]; seen = st.total.notna().to_numpy(); tot = st.total.to_numpy()
    crossed = seen & (tot >= 1400); u = seen & ~crossed
    r = dict(horizon_h=h, share_seen=seen.mean(), median_age_h=float(np.nanmedian(st.age)), share_already_past_1400=crossed.mean(),
             n_seen_not_past=int(u.sum()))
    for nm, sc in [("model", Sv[f"h{h}"].to_numpy()), ("pre", Sv.pre.to_numpy()), ("count_alone", tot)]:
        r[f"auc_not_past_{nm}"] = roc_auc_score(Sv["pop"].to_numpy()[u], sc[u])
    days = Sv.day.to_numpy()[u]; ud, inv = np.unique(days, return_inverse=True); bs = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))[inv]
        bs.append(roc_auc_score(Sv["pop"].to_numpy()[u], Sv[f"h{h}"].to_numpy()[u], sample_weight=w) - roc_auc_score(Sv["pop"].to_numpy()[u], tot[u], sample_weight=w))
    r["model_minus_count_lo"], r["model_minus_count_hi"] = np.quantile(bs, [.025, .975])
    r["auc_unseen_model"] = roc_auc_score(Sv["pop"].to_numpy()[~seen], Sv[f"h{h}"].to_numpy()[~seen])
    r["auc_unseen_pre"] = roc_auc_score(Sv["pop"].to_numpy()[~seen], Sv.pre.to_numpy()[~seen])
    info.append(r)
I = pd.DataFrame(info)
R.to_csv(T / "v8_first_hours.csv", index=False); D.to_csv(T / "v8_first_hours_diffs.csv", index=False); I.to_csv(T / "v8_first_hours_coverage.csv", index=False)
# which network: counts alone among articles sighted by 3 h
st = states[("", 3)].iloc[Sv.row]; seen = st.total.notna().to_numpy(); p = Sv["pop"].to_numpy()[seen]; top5 = (Sv.y.to_numpy() >= np.quantile(Sv.y, .95))[seen]
N = pd.DataFrame([dict(signal=k, auc_popular=roc_auc_score(p, st[k].to_numpy()[seen]), auc_top5pct=roc_auc_score(top5, st[k].to_numpy()[seen]),
                       mean_part_of_total=float(np.mean(np.expm1(st[k].to_numpy()[seen]) / (st.total.to_numpy()[seen] + 1))),
                       sd_log=float(np.std(st[k].to_numpy()[seen])))
                  for k in ["l_total", "l_twitter", "l_facebook", "l_linked_in", "l_google_plus"]])
N.to_csv(T / "v8_networks.csv", index=False)
pd.set_option("display.width", 220)
print(R.pivot(index="scorer", columns="metric", values="est").loc[MODELS].round(4))
print(D[D.metric.isin(["auc", "pair_sec"])].round(4).to_string(index=False))
print(I.round(4).to_string(index=False)); print(N.round(4).to_string(index=False))
