"""S1b: precision of slug twins against 180 annotated pairs, before and after the series filter.

Annotation (data/annotations/twins_sample.csv, column `label`) was done by the authors' assistant
(Claude), not by independent annotators, from slug + dates + channel only:
  S = same news story/event, R = recurring series instalment, T = same broad topic, different story,
  U = unrelated (lexical coincidence).
Output: results/tables/s1b_twin_precision.csv
"""
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from newspop.data import ROOT, load
from newspop import stories as st

df = load()
A = pd.read_csv(ROOT / "data" / "annotations" / "twins_sample.csv")
A["dt"] = (pd.to_datetime(A.date_j) - pd.to_datetime(A.date_i)).dt.days.abs()
A["series_rule"] = st.is_series(df, A)
print(pd.crosstab(A.label, A.series_rule))
rows = []
for name, sub in [("all bands >= 0.5", A[A.sim >= 0.5]), ("all bands >= 0.3", A)] + \
        [(f"band {b}", A[A.band == b]) for b in sorted(A.band.unique())]:
    for filt in [False, True]:
        S = sub[~sub.series_rule] if filt else sub
        k, n = int((S.label == "S").sum()), len(S)
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"subset": name, "series_filtered": filt, "n": n, "same_story": k,
                     "precision": round(k / n, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                     "share_same_topic_or_story": round(S.label.isin(["S", "T"]).mean(), 3)})
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / "s1b_twin_precision.csv", index=False)
print(R.to_string(index=False))
