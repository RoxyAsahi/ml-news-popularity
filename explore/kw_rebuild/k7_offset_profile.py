"""Outcome-based check of the re-alignment: shares played no part in k4_realign.py, so they can test it.

(Suggested and first run by the review thread, /mnt/project-files/review3/realign_outcome_check.py.)
A ridge on TF-IDF of the real headline is fit on file rows 0..4,599 (January to March 2013, where the file is
aligned) to predict log shares minus the day mean. For later rows the headline is taken from the url column at
offset k_i + d, d = -3..3, where k_i is the estimated offset. If k_i is right, the within-day correlation of the
prediction with log shares peaks at d = 0 and is about zero one row to either side.
Reported by period, for the rows within 5 rows of an offset step, and for the rows where neither title fingerprint
agrees. Also: monotonicity of the offset and the longest run of doubtful rows.
Output: results/tables/k7_offset_profile.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwlib import ACQUIRED, INT, TAB  # noqa: E402
from newspop.data import load_raw  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

raw = load_raw().reset_index(drop=True)
ra = pd.read_csv(INT / "realigned.csv")
assert (ra.url_in_file == raw.url).all()
n = len(raw); idx = np.arange(n); k = ra.offset.to_numpy().astype(int); inside = idx + k < n
title_at_row = aligned_titles(raw).fillna("").to_numpy()  # headline of the URL sitting in each file row
date = (ACQUIRED - pd.to_timedelta(raw.timedelta, unit="D")).to_numpy()
ly = np.log1p(raw.shares.to_numpy().astype(float))
dm = ly - pd.Series(ly).groupby(date).transform("mean").to_numpy()
tit = lambda off: title_at_row[np.clip(idx + off, 0, n - 1)]  # noqa: E731
train = idx < 4600
vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True)
model = Ridge(alpha=1.0).fit(vec.fit_transform(tit(0)[train]), dm[train])


def corr(off, mask):
    p = model.predict(vec.transform(tit(off)[mask]))
    return float(np.corrcoef(p - pd.Series(p).groupby(date[mask]).transform("mean").to_numpy(), dm[mask])[0, 1])


steps = np.flatnonzero(np.diff(k) > 0); near = np.zeros(n, bool)
for s in steps:
    near[max(0, s - 5): s + 6] = True
both = (~ra.token_count_agrees.astype(bool) & ~ra.sentiment_agrees.astype(bool)).to_numpy() & inside
d64 = np.datetime64
SETS = {"file rows 4,600..8,008 (aligned, held out)": (idx >= 4600) & (idx < 8009),
        "2013-06-06..2013-12-31": (idx >= 8009) & (date < d64("2014-01-01")),
        "2014 H1": (date >= d64("2014-01-01")) & (date < d64("2014-07-01")),
        "2014 H2": (date >= d64("2014-07-01")) & inside,
        "within 5 rows of an offset step": near & inside & (idx >= 8009),
        "neither fingerprint agrees (after 2013-06-06)": both & (idx >= 8009)}
rows = []
for name, m in SETS.items():
    r = dict(rows=name, n=int(m.sum()), as_distributed=corr(0, m))
    r.update({f"d={d:+d}": corr(k + d, m) for d in range(-3, 4)})
    r["token_count_agrees"], r["sentiment_agrees"] = ra.token_count_agrees[m].mean(), ra.sentiment_agrees[m].mean()
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(TAB / "k7_offset_profile.csv", index=False)
b = both.astype(int); runs, c = [0], 0
for x in b:
    c = c + 1 if x else 0; runs.append(c)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
print("offset non-decreasing:", bool((np.diff(k) >= 0).all()), "| steps:", len(steps), "| rows where neither fingerprint agrees:",
      int(both.sum()), f"({both.mean():.3%}; before the first shifted row {both[:8009].mean():.3%})", "| longest run:", max(runs))
