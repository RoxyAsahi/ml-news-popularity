"""Probe C: does outside attention to an article's topic, measured BEFORE publication, predict its shares?

Entities: word n-grams (1-4) of the real headline that are titles of English Wikipedia articles with at least 1,000
views in some month (longest match first; stop words and a short list of generic words are not matched alone).
Attention: daily Wikipedia pageviews (desktop + mobile) of the matched pages.
Ex-ante features for an article published on day d (only days < d are used):
  wk_n          number of matched entities
  wk_lv1        max over entities of log1p(views on d-1)
  wk_lv7        max over entities of log1p(mean views d-7..d-1)
  wk_surge1     max over entities of log1p(views d-1) - log1p(mean views d-37..d-8)
  wk_surge7     max over entities of log1p(mean d-7..d-1) - log1p(mean d-37..d-8)
  wk_surge_top  surge1 of the entity with the largest baseline;  wk_lbase  its log baseline
Evaluation, past-only monthly rolling origin from 2013-07:
  hgb (58 features)  vs  hgb + wk_*   -> AUC(>=1400), top-5% AUC, within-day Spearman, day-bootstrap differences;
  residual check: Spearman of the hgb residual with each wk feature.
Needs /home/user/work/wiki/pv_YYYYMM.tsv.gz (c1_fetch_wiki_pageviews.py). Months without a file are skipped,
so the script can run on a partial download (--until).
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HGB_PARAMS, ROOT, features, load_fixed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wiki", default="/home/user/work/wiki")
ap.add_argument("--first", default="2013-07")
ap.add_argument("--until", default="2014-12")
a = ap.parse_args()
GENERIC = set("""new video photos photo app apps watch best top day week year years time first world people way things
make big little good great free live life home work game games music movie film tv show news story stories guide
review reviews tips how why what who where when says say look looks get gets got just one two three four five six
seven eight nine ten million billion launch launches plan plans report reports deal back here now today tomorrow
out off up down like share shares social media tech business real change future look inside open start end right
set play run help need want know think love man woman men women kid kids girl boy dog cat baby car phone tablet
web internet online site sites company companies startup startups user users number numbers data picture pictures
image images infographic comic comics pics amazing awesome perfect ultimate happy better more most less least
latest early late next last old young small large high low hot cool cold dead alive pass passes win wins lose
makes making made take takes taken give gives turn turns come comes go goes going see sees seen find finds found
""".split())

df = load_fixed()
df = df[df.title.notna() & (df.date.dt.to_period("M").astype(str) <= a.until)].reset_index(drop=True)
n = len(df)
day0 = pd.Timestamp("2012-12-01")
dix = (df.date - day0).dt.days.to_numpy()
ND = (pd.Timestamp("2015-01-01") - day0).days

# ---------------------------------------------------------------- candidate n-grams from headlines
tok_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.&-]*")


def grams(title):
    w = [t.strip(".'") for t in tok_re.findall(title.replace("’", "'"))]
    w = [re.sub(r"'s$", "", t) for t in w if t]
    out = []
    for L in (4, 3, 2, 1):
        for i in range(len(w) - L + 1):
            g = w[i:i + L]
            if L == 1 and (g[0].lower() in ENGLISH_STOP_WORDS or g[0].lower() in GENERIC or len(g[0]) < 3
                           or g[0].isdigit()):
                continue
            if g[0].lower() in ENGLISH_STOP_WORDS or g[-1].lower() in ENGLISH_STOP_WORDS:
                continue
            out.append((i, L, "_".join(g).lower()))
    return out


cand = [grams(t) for t in df.title]
want = {g for c in cand for _, _, g in c}
print(f"articles {n:,}; distinct candidate n-grams {len(want):,}", flush=True)

# ---------------------------------------------------------------- pageviews for matched titles
months = [m for m in pd.period_range("2012-12", "2014-12", freq="M").strftime("%Y%m")
          if (Path(a.wiki) / f"pv_{m}.tsv.gz").exists()]
print("pageview months available:", months[0], "..", months[-1], len(months), flush=True)
key_ix: dict[str, int] = {}
rows_i, rows_d, rows_v = [], [], []
num_re = re.compile(r"([A-Z\[\\\]^_])(\d+)")
for m in months:
    off = (pd.Timestamp(m[:4] + "-" + m[4:] + "-01") - day0).days
    with gzip.open(Path(a.wiki) / f"pv_{m}.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 4 or ":" in p[0]:
                continue
            k = unquote(p[0]).lower()
            if k not in want:
                continue
            j = key_ix.setdefault(k, len(key_ix))
            for d, v in num_re.findall(p[3]):
                rows_i.append(j); rows_d.append(off + ord(d) - 65); rows_v.append(int(v))
    print("  read", m, "matched titles so far", len(key_ix), flush=True)
V = np.zeros((len(key_ix), ND + 1), np.float32)
np.add.at(V, (np.array(rows_i), np.clip(np.array(rows_d), 0, ND)), np.array(rows_v, np.float32))
have = np.zeros(ND + 1, bool)
for m in months:
    s0 = (pd.Timestamp(m[:4] + "-" + m[4:] + "-01") - day0).days
    have[s0:s0 + pd.Period(m[:4] + "-" + m[4:]).days_in_month] = True
C = np.cumsum(np.c_[np.zeros(len(V)), V], axis=1)  # C[:, t] = sum of days < t


def mean_views(j, lo, hi):  # mean over days lo..hi-1
    return (C[j, hi] - C[j, lo]) / max(hi - lo, 1)


F = np.full((n, 7), np.nan)
names = ["wk_n", "wk_lv1", "wk_lv7", "wk_surge1", "wk_surge7", "wk_surge_top", "wk_lbase"]
ent_of = []
for r in range(n):
    d = dix[r]
    used = np.zeros(64, bool); ents = []
    for i, L, g in cand[r]:  # longest first; do not reuse covered words
        j = key_ix.get(g)
        if j is None or used[i:i + L].any():
            continue
        used[i:i + L] = True; ents.append(j)
    ent_of.append(ents)
    if d < 38 or not have[d - 37:d].all():
        continue
    F[r, 0] = len(ents)
    if not ents:
        F[r, 1:] = 0
        continue
    v1 = np.array([V[j, d - 1] for j in ents]); v7 = np.array([mean_views(j, d - 7, d) for j in ents])
    base = np.array([mean_views(j, d - 37, d - 7) for j in ents])
    s1 = np.log1p(v1) - np.log1p(base); s7 = np.log1p(v7) - np.log1p(base)
    t = int(np.argmax(base))
    F[r, 1:] = [np.log1p(v1).max(), np.log1p(v7).max(), s1.max(), s7.max(), s1[t], np.log1p(base[t])]
W = pd.DataFrame(F, columns=names)
print("articles with >=1 entity:", float((W.wk_n > 0).mean().round(3)), "mean entities", float(W.wk_n.mean().round(2)))
inv = {v: k for k, v in key_ix.items()}
for r in np.random.default_rng(0).choice(n, 12, replace=False):
    print("   ", df.title[r][:70], "->", [inv[j] for j in ent_of[r]])

# ---------------------------------------------------------------- evaluation
X = features(df)
y = np.log1p(df.shares.to_numpy().astype(float)); s = df.shares.to_numpy()
month = df.date.dt.to_period("M").astype(str).to_numpy()
t95 = np.quantile(s, .95)
sc = {k: np.full(n, np.nan) for k in ["hgb", "hgb_wk"]}
XW = np.c_[X.to_numpy(), W.to_numpy()]
for m in sorted(set(month)):
    if m < a.first:
        continue
    tr, te = month < m, month == m
    sc["hgb"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(X.to_numpy()[tr], y[tr]).predict(X.to_numpy()[te])
    sc["hgb_wk"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(XW[tr], y[tr]).predict(XW[te])
    print(m, "done", flush=True)
ev = ~np.isnan(sc["hgb"])
dd = dix[ev]; days = np.unique(dd); gi = [np.flatnonzero(dd == d) for d in days]
rng = np.random.default_rng(0); boot = rng.integers(0, len(days), size=(300, len(days)))


def stats(z, sub=None):
    ss, yy = (s[ev], y[ev]) if sub is None else (s[ev][sub], y[ev][sub])
    zz = z if sub is None else z[sub]
    return np.array([roc_auc_score(ss >= 1400, zz), roc_auc_score(ss >= t95, zz), spearmanr(zz, yy)[0]])


def rho_day(z):
    out = []
    for ix in gi:
        if len(ix) > 3:
            out.append(np.corrcoef(rankdata(z[ix]), rankdata(y[ev][ix]))[0, 1])
    return float(np.nanmean(out))


res = {k: stats(sc[k][ev]) for k in sc}
for k in sc:
    print(f"{k:8s} auc1400 {res[k][0]:.4f}  auc_top5 {res[k][1]:.4f}  spearman {res[k][2]:.4f}  rho_day {rho_day(sc[k][ev]):.4f}")
diffs = []
for b in boot:
    sub = np.concatenate([gi[j] for j in b])
    diffs.append(stats(sc["hgb_wk"][ev], sub) - stats(sc["hgb"][ev], sub))
D = np.array(diffs)
for j, nm in enumerate(["auc1400", "auc_top5", "spearman"]):
    print(f"diff {nm}: {res['hgb_wk'][j] - res['hgb'][j]:+.4f} [{np.quantile(D[:, j], .025):+.4f}, {np.quantile(D[:, j], .975):+.4f}]")
resid = y[ev] - sc["hgb"][ev]
for nm in names:
    v = W[nm].to_numpy()[ev]; ok = ~np.isnan(v)
    print(f"residual vs {nm:13s} spearman {spearmanr(v[ok], resid[ok])[0]:+.4f}   raw vs log shares {spearmanr(v[ok], y[ev][ok])[0]:+.4f}")
has = W.wk_n.to_numpy()[ev] > 0
q = pd.qcut(W.wk_surge1.to_numpy()[ev][has], 5, labels=False, duplicates="drop")
print(pd.DataFrame({"q": q, "resid": resid[has], "y": y[ev][has]}).groupby("q").agg(["mean", "size"]).round(3).to_string())
out = ROOT / "data" / "interim"; out.mkdir(parents=True, exist_ok=True)
W.assign(file_row=df.file_row).to_parquet(out / "c2_wiki_features.parquet", index=False)
