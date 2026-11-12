"""Same-story "twin" articles from URL slugs, plus context helpers (channel, day, week).

Mashable URLs end in a slug that is a shortened headline, e.g.
http://mashable.com/2013/01/07/amazon-instant-video-browser/ -> "amazon instant video browser".
The slug is the only text the dataset carries. Two articles whose slugs are close in TF-IDF space
and that were published a few days apart are usually the same news story ("twins"). Pairs of twins
give a topic- and time-controlled comparison (in the spirit of Tan, Lee & Pang, ACL 2014).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

CHANNELS = ["lifestyle", "entertainment", "bus", "socmed", "tech", "world"]

# defaults used everywhere downstream (chosen in scripts/s1_twin_precision.py)
SIM_THRESHOLD = 0.5
MAX_DAYS = 7


def slug_text(urls: pd.Series) -> pd.Series:
    """URL -> slug words. Drops the trailing '-2', '-3' Mashable adds to repeated slugs."""
    slug = urls.str.extract(r"/\d{4}/\d{2}/\d{2}/([^/]+)/?\s*$")[0]
    slug = slug.str.replace(r"-\d+$", "", regex=True)
    return slug.str.replace("-", " ").str.strip()


def channel(df: pd.DataFrame) -> np.ndarray:
    """Channel name per article; 'other' when no data_channel_is_* flag is set."""
    flags = df[[f"data_channel_is_{c}" for c in CHANNELS]].to_numpy()
    out = np.array(CHANNELS, dtype=object)[flags.argmax(1)]
    out[flags.sum(1) == 0] = "other"
    return out


def day_index(df: pd.DataFrame) -> np.ndarray:
    """Days since the first publication date in the data."""
    return (df["publish_date"] - df["publish_date"].min()).dt.days.to_numpy()


def week_index(df: pd.DataFrame) -> np.ndarray:
    return day_index(df) // 7


def slug_vectors(df: pd.DataFrame):
    """L2-normalised TF-IDF vectors of slug words (unigrams + bigrams, sublinear tf)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    return vec.fit_transform(slug_text(df["url"]))


def twin_pairs(df: pd.DataFrame, min_sim: float = 0.3, max_days: int = MAX_DAYS,
               chunk: int = 3000) -> pd.DataFrame:
    """All article pairs (i < j, row indices of df) with slug cosine >= min_sim and
    publication dates at most max_days apart. Returns columns i, j, sim, dt, same_channel."""
    V = slug_vectors(df)
    day = day_index(df)
    chan = channel(df)
    rows = []
    n = V.shape[0]
    for s in range(0, n, chunk):
        S = (V[s:s + chunk] @ V.T).tocoo()
        i = S.row + s
        keep = (S.data >= min_sim) & (S.col > i) & (np.abs(day[i] - day[S.col]) <= max_days)
        rows.append(pd.DataFrame({"i": i[keep], "j": S.col[keep], "sim": S.data[keep]}))
    P = pd.concat(rows, ignore_index=True)
    P["sim"] = P["sim"].clip(upper=1.0)
    P["dt"] = np.abs(day[P.i] - day[P.j])
    P["same_channel"] = chan[P.i] == chan[P.j]
    return P.sort_values(["i", "j"]).reset_index(drop=True)


def story_ids(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Story label per article = connected component of the twin graph (singletons get their own)."""
    A = coo_matrix((np.ones(len(pairs)), (pairs.i, pairs.j)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return lab


# ------------------------------------------------------------------ recurring series filter
_ORDINALS = set(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "first second third fourth fifth monday tuesday wednesday thursday friday saturday sunday "
    "a b c d e f g h part".split())


def _tail_number(url: str) -> int:
    m = re.search(r"-(\d+)/?\s*$", url)
    return int(m.group(1)) if m else 0


def is_series(df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """True for pairs that are instalments of a recurring column rather than one news story
    (e.g. 'viral video recap', 'top moments olympics day four' vs 'day eight', episode recaps).

    Rules (validated on the 180 annotated pairs: catches 27 of 33 series, 0 false positives):
      - identical slugs that recur >= 4 times in the data, or are >= 5 days apart, or carry
        Mashable's numeric suffix >= 3;
      - one slug equals a slug that recurs >= 4 times and is contained in the other;
      - both slugs contain 'recap';
      - the slugs differ only by numbers, ordinals, weekdays or single letters.
    """
    s = slug_text(df["url"]).to_numpy()
    urls = df["url"].to_numpy()
    cnt = pd.Series(s).value_counts()
    out = np.zeros(len(pairs), bool)
    for k, (i, j, dt) in enumerate(zip(pairs.i.to_numpy(), pairs.j.to_numpy(), pairs.dt.to_numpy())):
        a, b = s[i], s[j]
        A, B = a.split(), b.split()
        if a == b:
            out[k] = cnt[a] >= 4 or dt >= 5 or max(_tail_number(urls[i]), _tail_number(urls[j])) >= 3
        elif (cnt[a] >= 4 and a in b) or (cnt[b] >= 4 and b in a):
            out[k] = True
        elif "recap" in A and "recap" in B:
            out[k] = True
        else:
            diff = set(A) ^ set(B)
            out[k] = bool(diff) and all(t.isdigit() or t in _ORDINALS for t in diff)
    return out


def clean_twins(df: pd.DataFrame, min_sim: float = SIM_THRESHOLD, max_days: int = MAX_DAYS) -> pd.DataFrame:
    """Twin pairs used downstream: similarity >= min_sim, <= max_days apart, series removed.
    Adds `first` = row index of the earlier article (ties: lower row index)."""
    P = twin_pairs(df, min_sim=min_sim, max_days=max_days)
    P = P[~is_series(df, P)].reset_index(drop=True)
    day = day_index(df)
    P["first"] = np.where(day[P.i] <= day[P.j], P.i, P.j)
    return P
