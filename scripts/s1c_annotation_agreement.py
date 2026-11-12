"""S1c: agreement between two independent large-language-model annotations of twin pairs.

Annotator 1 = the AI assistant that labelled all 180 pairs (data/annotations/twins_sample.csv).
Annotator 2 = a separate AI (LLM) session that saw only slugs, dates and channels of 60 of those pairs
(10 per similarity band, shuffled) and none of annotator 1's labels
(data/annotations/twins_blind60_annotator2.csv). Neither annotator is human.
Reports raw agreement and Cohen's kappa on the four labels (S/R/T/U) and on same-story vs not, and the
same-story precision (sim >= 0.5, series-filtered) under annotator 2.

Output: results/tables/s1c_annotation_agreement.csv
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from newspop.data import ROOT, load
from newspop import stories as st

A = pd.read_csv(ROOT / "data" / "annotations" / "twins_sample.csv")
B = pd.read_csv(ROOT / "data" / "annotations" / "twins_blind60_annotator2.csv")[["pair_id", "label"]]
M = A.merge(B, on="pair_id", suffixes=("_1", "_2"))
assert len(M) == 60
M["dt"] = (pd.to_datetime(M.date_j) - pd.to_datetime(M.date_i)).dt.days.abs()
M["series_rule"] = st.is_series(load(), M)
print(pd.crosstab(M.label_1, M.label_2))
s1, s2 = M.label_1 == "S", M.label_2 == "S"
F = M[(M.sim >= 0.5) & ~M.series_rule]
rows = [
    dict(measure="raw agreement, 4 labels", value=(M.label_1 == M.label_2).mean(), n=len(M)),
    dict(measure="Cohen kappa, 4 labels", value=cohen_kappa_score(M.label_1, M.label_2), n=len(M)),
    dict(measure="raw agreement, same story vs not", value=(s1 == s2).mean(), n=len(M)),
    dict(measure="Cohen kappa, same story vs not", value=cohen_kappa_score(s1, s2), n=len(M)),
    dict(measure="same-story precision sim>=0.5 filtered, annotator 1", value=(F.label_1 == "S").mean(), n=len(F)),
    dict(measure="same-story precision sim>=0.5 filtered, annotator 2", value=(F.label_2 == "S").mean(), n=len(F)),
]
R = pd.DataFrame(rows).round(3)
R.to_csv(ROOT / "results" / "tables" / "s1c_annotation_agreement.csv", index=False)
print(R.to_string(index=False))
