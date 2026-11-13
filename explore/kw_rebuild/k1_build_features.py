"""Rebuild keyword-history ("memory") features as a stream, from keys we can actually see.

The dataset's nine kw_* columns are statistics of the shares of EARLIER articles that carry the same Mashable
meta keyword (editor tags). The tags themselves are not in the data. Here the same kind of memory is rebuilt
from keys that are visible: URL-slug words, words and bigrams of the real headline, and the section.

For article i published on day t and a key k, the memory is computed from the articles with key k published on
the W days that end at day t - L, i.e. days in [t - L - W + 1, t - L]:
  L (lag, "embargo")  1 = strictly earlier days (default); 3, 7, 14, 30 = only articles at least L days old,
                      whose share counts were (nearly) final by day t. Shares are still the January-2015 crawl
                      values, so L = 1 inherits the same hindsight as the original kw_* columns.
  W (window)          7, 30, 90 days or all history; plus exponentially decayed versions (half-life 7, 30, 90).

Per (article, key): n, mean of y = log1p(shares), and an empirical-Bayes lift
      theta = n / (n + m) * (mean_y - mu)
where mu is the mean of y over ALL articles in the same window (so the lift is free of the site-wide level) and
m = sigma^2 / tau^2 is estimated by the method of moments on the burn-in months only (before 2013-07):
tau^2 = mean of (y_i - mu)(mean_y_k - mu) over (article, key) pairs, sigma^2 = var(y_i - mu).
Per article the key lifts are aggregated (mean, max, min, sum, precision-weighted mean) with support counts.

Families written (columns are prefixed <source>_):
  mim   "mimic": the original recipe on visible keys - min/avg/max over keys of the min/avg/max raw shares of
        earlier articles, all history, no shrinkage (9 columns)
  all   EB lifts, all history
  win   EB lifts for windows 7 / 30 / 90 days
  dec   EB lifts with exponential decay, half-life 7 / 30 / 90 days
  mom   momentum: short-window lift minus all-history lift, and share of the key's history that is recent
Sources: slug (URL slug words), tit (headline unigrams + bigrams), sec (section; a single key per article).

Alignment (first argument, default "true"): the url column of the CSV is shifted against the other columns from
June 2013 on (k4_realign.py). "given" builds the keys from the url column as distributed, "true" from the
re-aligned URL / headline and the timedelta date. Run k4_realign.py first.

Output: data/interim/kw_rebuild/features_{align}_L{lag}.parquet (one row per article, order of kwlib.frame(align))
        and results/tables/k1_eb_strength.csv (estimated m per alignment / source / window).
Resumable: an existing features file is not rebuilt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwlib import frame  # noqa: E402
from newspop.stories import channel, slug_text  # noqa: E402

OUT = ROOT / "data" / "interim" / "kw_rebuild"
TAB = ROOT / "results" / "tables"
LAGS = [1, 3, 7, 14, 30]
WINDOWS = {"w7": 7, "w30": 30, "w90": 90, "all": None}
HALF = {"h7": 7.0, "h30": 30.0, "h90": 90.0}
BURN_END = "2013-07-01"
BIG = 100_000
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def keys_for(df: pd.DataFrame) -> dict[str, list[list[str]]]:
    slug = slug_text(df.url.fillna("")).fillna("")
    title = df.title.where(df.title.notna(), slug)  # unmatched headlines fall back to the slug
    out = {"slug": [], "tit": [], "sec": []}
    for s, t, c in zip(slug, title, channel(df)):
        out["slug"].append(sorted(set(tokens(s))))
        tk = tokens(t)
        out["tit"].append(sorted(set(tk) | {a + "_" + b for a, b in zip(tk, tk[1:])}))
        out["sec"].append([c])
    out["ALL"] = [["all"]] * len(df)
    return out


class Stream:
    """Long (article, key) table sorted by key then day, with prefix sums that answer window queries."""

    def __init__(self, keys: list[list[str]], day: np.ndarray, y: np.ndarray, shares: np.ndarray):
        art = np.repeat(np.arange(len(keys)), [len(k) for k in keys])
        kid = pd.factorize(np.array([k for ks in keys for k in ks], dtype=object))[0]
        o = np.lexsort((day[art], kid))
        self.art, self.kid, self.d = art[o], kid[o], day[art][o]
        self.comp = self.kid.astype(np.int64) * BIG + self.d
        self.y, self.sh = y[self.art], shares[self.art]
        self.start = np.searchsorted(self.comp, self.kid.astype(np.int64) * BIG - BIG // 2, "left")
        self.cy = np.r_[0.0, np.cumsum(self.y)]
        self.csh = np.r_[0.0, np.cumsum(self.sh)]
        g = pd.Series(self.sh).groupby(self.kid)
        self.cmax, self.cmin = g.cummax().to_numpy(), g.cummin().to_numpy()
        self.dec = {}
        for name, h in HALF.items():
            w = np.exp2(self.d / h)
            gw = pd.DataFrame({"w": w, "wy": w * self.y}).groupby(self.kid).cumsum()
            self.dec[name] = (gw.w.to_numpy(), gw.wy.to_numpy())

    def hi(self, lag: int) -> np.ndarray:
        """Exclusive end position of the occurrences of the same key with day <= t - lag."""
        return np.searchsorted(self.comp, self.comp - lag, "right")

    def window(self, lag: int, W: int | None):
        hi = self.hi(lag)
        lo = self.start if W is None else np.maximum(np.searchsorted(self.comp, self.comp - W - lag + 1, "left"), self.start)
        lo = np.minimum(lo, hi)
        n = (hi - lo).astype(float)
        return n, self.cy[hi] - self.cy[lo]

    def decayed(self, lag: int, name: str):
        hi = self.hi(lag); ok = hi > self.start
        cw, cwy = self.dec[name]
        f = np.exp2(-self.d / HALF[name])
        n = np.where(ok, cw[np.maximum(hi - 1, 0)] * f, 0.0)
        s = np.where(ok, cwy[np.maximum(hi - 1, 0)] * f, 0.0)
        return n, s

    def raw(self, lag: int):
        """All-history raw-share stats per (article, key): n, min, max, mean (NaN when no history)."""
        hi = self.hi(lag); ok = hi > self.start; j = np.maximum(hi - 1, 0)
        n = (hi - self.start).astype(float)
        mean = np.where(ok, (self.csh[hi] - self.csh[self.start]) / np.maximum(n, 1), np.nan)
        return n, np.where(ok, self.cmin[j], np.nan), np.where(ok, self.cmax[j], np.nan), mean


def per_article(st: Stream, values: np.ndarray, how: str, n_art: int, w: np.ndarray | None = None) -> np.ndarray:
    ok = ~np.isnan(values)
    s = pd.Series(values[ok]); g = st.art[ok]
    if how == "wmean":
        num = np.bincount(g, weights=values[ok] * w[ok], minlength=n_art); den = np.bincount(g, weights=w[ok], minlength=n_art)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    r = getattr(s.groupby(g), how)()
    out = np.full(n_art, np.nan); out[r.index.to_numpy()] = r.to_numpy()
    return out


def build(lag: int, df: pd.DataFrame, keys: dict, day: np.ndarray, y: np.ndarray, burn: np.ndarray):
    n_art = len(df); shares = df.shares.to_numpy().astype(float)
    glob = Stream(keys["ALL"], day, y, shares)
    assert np.array_equal(glob.art, np.arange(n_art))  # df is sorted by date
    mu = {}
    for wn, W in WINDOWS.items():
        n, s = glob.window(lag, W); mu[wn] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    for hn in HALF:
        n, s = glob.decayed(lag, hn); mu[hn] = np.where(n > 0, s / np.maximum(n, 1e-12), np.nan)
    F, strength = {}, []
    for src in ["slug", "tit", "sec"]:
        st = Stream(keys[src], day, y, shares)
        yi = st.y
        theta = {}
        for name in list(WINDOWS) + list(HALF):
            n, s = st.window(lag, WINDOWS[name]) if name in WINDOWS else st.decayed(lag, name)
            has = n > 1e-9
            m_art = mu[name][st.art]
            dev = np.where(has, s / np.maximum(n, 1e-12) - m_art, np.nan)
            b = has & burn[st.art] & ~np.isnan(m_art)
            tau2 = float(np.mean((yi[b] - m_art[b]) * dev[b])) if b.sum() > 50 else 0.0
            sig2 = float(np.var(yi[b] - m_art[b])) if b.sum() > 50 else 1.0
            m = sig2 / tau2 if tau2 > 1e-4 else 1e4  # no between-key variance detected -> shrink (almost) fully
            strength.append(dict(lag=lag, source=src, window=name, pairs=int(b.sum()), tau2=tau2, sigma2=sig2, m=m))
            th = np.where(has, n / (n + m) * dev, np.nan)
            theta[name] = (th, np.where(has, n, np.nan))
            fam = "all" if name == "all" else ("win" if name in WINDOWS else "dec")
            if src == "sec":
                F[f"sec_{fam}_{name}_dev"] = per_article(st, dev, "mean", n_art)
                continue
            p = f"{src}_{fam}_{name}"
            F[p + "_mean"] = per_article(st, th, "mean", n_art)
            F[p + "_max"] = per_article(st, th, "max", n_art)
            F[p + "_min"] = per_article(st, th, "min", n_art)
            F[p + "_sum"] = per_article(st, th, "sum", n_art)
            F[p + "_nkeys"] = np.nan_to_num(per_article(st, np.where(has, 1.0, np.nan), "sum", n_art))
            F[p + "_lognmax"] = np.log1p(np.nan_to_num(per_article(st, np.where(has, n, np.nan), "max", n_art)))
        if src == "sec":
            continue
        # momentum: is the key hotter (and busier) lately than over its whole history?
        for short in ["w7", "w30"]:
            d = theta[short][0] - theta["all"][0]
            F[f"{src}_mom_{short}_minus_all_mean"] = per_article(st, d, "mean", n_art)
            F[f"{src}_mom_{short}_minus_all_max"] = per_article(st, d, "max", n_art)
            frac = theta[short][1] / theta["all"][1]
            F[f"{src}_mom_{short}_recent_share_max"] = per_article(st, frac, "max", n_art)
        # mimic of the original recipe: raw shares, all history, no shrinkage
        n, mn, mx, av = st.raw(lag)
        for stat, v in [("min", mn), ("max", mx), ("avg", av)]:
            F[f"{src}_mim_min_{stat}"] = per_article(st, v, "min", n_art)
            F[f"{src}_mim_max_{stat}"] = per_article(st, v, "max", n_art)
            F[f"{src}_mim_avg_{stat}"] = per_article(st, v, "mean", n_art)
    return pd.DataFrame(F), strength


def main():
    align = sys.argv[1] if len(sys.argv) > 1 else "true"  # "given" = url column as distributed, "true" = re-aligned
    OUT.mkdir(parents=True, exist_ok=True)
    df = frame(align)
    day = (df.publish_date - df.publish_date.min()).dt.days.to_numpy().astype(np.int64)
    y = np.log1p(df.shares.to_numpy().astype(float))
    burn = (df.publish_date < BURN_END).to_numpy()
    keys = keys_for(df)
    for s in ["slug", "tit"]:
        k = keys[s]
        print(s, "keys per article", np.mean([len(x) for x in k]).round(2), "distinct", len({a for x in k for a in x}), flush=True)
    rows = []
    for lag in LAGS:
        f = OUT / f"features_{align}_L{lag}.parquet"
        if f.exists():
            print("exists", f.name, flush=True)
            continue
        F, strength = build(lag, df, keys, day, y, burn)
        F.insert(0, "file_row", df.file_row.to_numpy())
        F.to_parquet(f); rows += [dict(align=align, **r) for r in strength]
        print("built", f.name, F.shape, flush=True)
    if rows:
        s = pd.DataFrame(rows)
        p = TAB / "k1_eb_strength.csv"
        if p.exists():
            s = pd.concat([pd.read_csv(p, dtype={"align": str}), s]).drop_duplicates(["align", "lag", "source", "window"], keep="last")
        s.sort_values(["align", "lag", "source", "window"]).to_csv(p, index=False)
        print(s[s.lag == s.lag.min()].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
