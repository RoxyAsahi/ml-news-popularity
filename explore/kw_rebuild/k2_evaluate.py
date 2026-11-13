"""Does a keyword memory rebuilt from visible keys replace (or add to) the dataset's kw_* columns?

Protocol = v19 (scripts/v1_past_only_called.py): monthly rolling origin, HistGB with newspop.anatomy.HGB_PARAMS on
log1p(shares); for each month m, fit on all articles published before m and score the articles of m.
Scores are produced for every month from 2013-03, and metrics are reported on two windows:
  v19   2013-07-01 .. 2014-12-31   (the v19 evaluation window)
  pre   2013-03-01 .. 2013-06-05   (before the first shifted row of the CSV, see k4_realign.py)

Alignment (first argument, default "true"): "given" uses the url column as distributed for headline, slug and
date, exactly as v19 did (its orig_all row reproduces v19's hgb row); "true" uses the re-aligned URL / headline and
the timedelta date (kwlib.frame). Dataset columns and shares are identical in both; only keys and dates differ.
Metrics: AUC (popular = shares >= 1400), mean within-day Spearman, share of days whose top-scored article is in
the day's top 10%, accuracy on same-day same-section pairs and on same-day pairs. 95% intervals and paired
differences: bootstrap over publication days (1,000 draws, same draws for every feature set).

Feature sets (see k1_build_features.py for the rebuilt families; lag 1 unless stated):
  orig_all          the 58 dataset features (+ 2 cleaning flags), = v19 "hgb"
  no_kw             without kw_* (9 columns + the -1 flag)
  no_kw+mim         + mimic of the original recipe on slug / headline words
  no_kw+words       + all word-based families (mim, all, win, dec, mom; slug and headline)
  no_kw+sec         + section memory
  no_kw+full        + words + section memory
  no_kw+full_L{k}   the same with an embargo of k days (only articles at least k days old feed the memory)
  orig_all+...      original columns plus rebuilt families (mim / all / all+win / all+dec / full)
  kw_only, full_only, no_hist (no kw_*, no self_reference_*), no_hist+full
  tfidf_ridge       ridge on TF-IDF of the headline (v19's text baseline)
  ...+judge         (true alignment only) + the cached zero-shot JEV lineup score of the headline (k5_rescore_judge.py)
Resumable: scores are cached per feature set and month in data/interim/kw_rebuild/scores_{align}/.
Output: results/tables/k2_metrics_{align}.csv, results/tables/k2_diffs_{align}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwlib import INT, TAB, frame, numeric_features  # noqa: E402
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.stories import channel  # noqa: E402

ALIGN = sys.argv[1] if len(sys.argv) > 1 else "true"
SC = INT / f"scores_{ALIGN}"
FIRST, B = "2013-03", 1000
WINDOWS = {"v19": ("2013-07-01", "2014-12-31"), "pre": ("2013-03-01", "2013-06-05")}

df = frame(ALIGN)
n = len(df)
y = np.log1p(df.shares.to_numpy()); pop = (df.shares.to_numpy() >= 1400).astype(int)
X = numeric_features(df)
cols = list(X.columns)
KW = [c for c in cols if c.startswith("kw_")]
REF = [c for c in cols if c.startswith("self_reference_")]
NO_KW = [c for c in cols if c not in KW]
day = pd.to_datetime(df.publish_date).dt.normalize()
dn = (day - day.min()).dt.days.to_numpy()
month = day.dt.to_period("M").astype(str).to_numpy()
chan = channel(df)
has_title = df.title.notna().to_numpy()  # as in v19, rows without a headline are left out of every metric

R = {L: pd.read_parquet(INT / f"features_{ALIGN}_L{L}.parquet").drop(columns="file_row") for L in [1, 3, 7, 14, 30]}
rc = list(R[1].columns)


def fam(*names, src=("slug", "tit")):
    return [c for c in rc if c.split("_")[0] in src and c.split("_")[1] in names]


WORDS = fam("mim", "all", "win", "dec", "mom")
SEC = [c for c in rc if c.startswith("sec_")]
FULL = WORDS + SEC
SETS = {
    "orig_all": (cols, [], 1),
    "no_kw": (NO_KW, [], 1),
    "no_kw+mim": (NO_KW, fam("mim"), 1),
    "no_kw+words": (NO_KW, WORDS, 1),
    "no_kw+sec": (NO_KW, SEC, 1),
    "no_kw+full": (NO_KW, FULL, 1),
    "orig_all+full": (cols, FULL, 1),
    "orig_all+mim": (cols, fam("mim"), 1),
    "orig_all+all": (cols, fam("all"), 1),
    "orig_all+all+win": (cols, fam("all", "win"), 1),
    "orig_all+all+dec": (cols, fam("all", "dec"), 1),
    "orig_all+words_slug": (cols, fam("mim", "all", "win", "dec", "mom", src=("slug",)), 1),
    "orig_all+words_tit": (cols, fam("mim", "all", "win", "dec", "mom", src=("tit",)), 1),
    "kw_only": (KW, [], 1),
    "full_only": ([], FULL, 1),
    "no_hist": ([c for c in NO_KW if c not in REF], [], 1),
    "no_hist+full": ([c for c in NO_KW if c not in REF], FULL, 1),
    **{f"no_kw+full_L{L}": (NO_KW, FULL, L) for L in [3, 7, 14, 30]},
    **{f"orig_all+full_L{L}": (cols, FULL, L) for L in [7, 30]},
}
JUDGE = INT / "judge_lineup_true.csv"  # k5_rescore_judge.py: zero-shot JEV score of each headline within its day's lineup
if ALIGN == "true" and JUDGE.exists():
    jl = pd.read_csv(JUDGE).drop_duplicates("file_row").set_index("file_row").p_top
    R[1]["judge_p_top"] = jl.reindex(df.file_row.to_numpy()).to_numpy()
    SETS.update({"orig_all+judge": (cols, ["judge_p_top"], 1), "orig_all+full+judge": (cols, FULL + ["judge_p_top"], 1),
                 "no_kw+judge": (NO_KW, ["judge_p_top"], 1)})
only = sys.argv[2:]  # optional: names of sets to fit in this process (for running several processes)


def design(name):
    base, extra, L = SETS[name]
    return pd.concat([X[base], R[L][extra]], axis=1).to_numpy(dtype=float)


months = [m for m in sorted(set(month)) if m >= FIRST]
SC.mkdir(parents=True, exist_ok=True)
for name in SETS:
    if only and name not in only:
        continue
    f = SC / f"{name}.npy"
    s = np.load(f) if f.exists() else np.full(n, np.nan)
    todo = [m for m in months if np.isnan(s[month == m]).any()]
    if not todo:
        continue
    D = design(name)
    for m in todo:
        tr, te = month < m, month == m
        s[te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(D[tr], y[tr]).predict(D[te])
        np.save(f, s)
    print("scored", name, D.shape[1], "features", flush=True)
# text baseline of v19: ridge on TF-IDF of the headline, same rolling origin
f = SC / "tfidf_ridge.npy"
if not only or "tfidf_ridge" in only:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    s = np.load(f) if f.exists() else np.full(n, np.nan)
    tt = df.title.fillna("")
    for m in [m for m in months if np.isnan(s[month == m]).any()]:
        tr, te = month < m, month == m
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(tt[tr])
        s[te] = Ridge(alpha=10).fit(v.transform(tt[tr]), y[tr]).predict(v.transform(tt[te]))
        np.save(f, s)
if only:
    sys.exit(0)

scores = {name: np.load(SC / f"{name}.npy") for name in list(SETS) + ["tfidf_ridge"] if (SC / f"{name}.npy").exists()}
scores["kw_avg_avg (no fit)"] = df.kw_avg_avg.to_numpy().astype(float)
for c in ["tit_all_all_mean", "slug_all_all_mean", "sec_win_w30_dev", "judge_p_top"]:
    if c in R[1]:
        scores[f"{c} (no fit)"] = np.nan_to_num(R[1][c].to_numpy())
MODELS = list(scores)


def pair_counts(s, yy, groups):
    c = t = 0.0
    for idx in groups:
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        dy = np.sign(yy[idx[a]] - yy[idx[b]]); ds = np.sign(s[idx[a]] - s[idx[b]]); k = dy != 0
        c += np.sum((dy[k] == ds[k]) + 0.5 * (ds[k] == 0)); t += k.sum()
    return c, t


rows, diffs = [], []
for wname, (lo, hi) in WINDOWS.items():
    ev = np.flatnonzero((day >= lo).to_numpy() & (day <= hi).to_numpy() & has_title)
    days = np.unique(dn[ev]); D = len(days); di = np.searchsorted(days, dn[ev])
    yv, pv, cv = y[ev], pop[ev], chan[ev]
    day_rows = [np.flatnonzero(di == d) for d in range(D)]
    per = {}
    for mname in MODELS:
        s = scores[mname][ev]
        rho, top, cs, ts, cd, td = (np.full(D, np.nan) for _ in range(6))
        for d, idx in enumerate(day_rows):
            if len(idx) >= 10:
                rho[d] = np.corrcoef(rankdata(s[idx]), rankdata(yv[idx]))[0, 1]
                top[d] = float(yv[idx][np.argmax(s[idx])] >= np.quantile(yv[idx], 0.9))
            cs[d], ts[d] = pair_counts(s, yv, [idx[cv[idx] == c] for c in np.unique(cv[idx])])
            cd[d], td[d] = pair_counts(s, yv, [idx])
        per[mname] = (s, rho, top, cs, ts, cd, td)
    base_top = np.nanmean([np.mean(yv[i] >= np.quantile(yv[i], 0.9)) if len(i) >= 10 else np.nan for i in day_rows])

    def stats(mname, w):
        s, rho, top, cs, ts, cd, td = per[mname]; ok = ~np.isnan(rho)
        return dict(auc=roc_auc_score(pv, s, sample_weight=w[di]),
                    rho_day=np.sum(w[ok] * rho[ok]) / np.sum(w[ok]), top10=np.sum(w[ok] * top[ok]) / np.sum(w[ok]),
                    pair_sec=np.sum(w * cs) / np.sum(w * ts), pair_day=np.sum(w * cd) / np.sum(w * td))

    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(B)]
    point = {m: stats(m, np.ones(D)) for m in MODELS}
    boot = {m: pd.DataFrame([stats(m, w) for w in W]) for m in MODELS}
    for m in MODELS:
        for k, v in point[m].items():
            rows.append(dict(window=wname, scorer=m, metric=k, est=v, lo=boot[m][k].quantile(.025), hi=boot[m][k].quantile(.975)))
        for ref in ["orig_all", "no_kw"]:
            if m == ref:
                continue
            bd = boot[m] - boot[ref]
            for k in point[m]:
                diffs.append(dict(window=wname, scorer=m, vs=ref, metric=k, diff=point[m][k] - point[ref][k],
                                  lo=bd[k].quantile(.025), hi=bd[k].quantile(.975)))
    rows.append(dict(window=wname, scorer="random", metric="top10", est=base_top, lo=np.nan, hi=np.nan))
    print(wname, "articles", len(ev), "days", D, flush=True)

Rm = pd.DataFrame(rows); Dm = pd.DataFrame(diffs)
Rm.insert(0, "align", ALIGN); Dm.insert(0, "align", ALIGN)
Rm.to_csv(TAB / f"k2_metrics_{ALIGN}.csv", index=False); Dm.to_csv(TAB / f"k2_diffs_{ALIGN}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for wname in WINDOWS:
    print("\n==", wname)
    print(Rm[Rm.window == wname].pivot(index="scorer", columns="metric", values="est").reindex(MODELS + ["random"]).round(4).to_string())
