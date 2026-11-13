"""Independent check of the re-alignment (k4_realign.py), using nothing the alignment itself looked at.

The alignment used the date, the title token count and the title sentiment. Here the check is the SECTION: a
classifier learns to tell the section (data_channel_is_*) from the URL-slug words on the rows before the first
shifted row, and is then scored on later rows, pairing each row's section flags with
  (a) the slug of the url column as distributed,   (b) the slug of the re-aligned URL.
If the url column were fine, (a) would stay near the held-out accuracy of the early period; if the re-alignment is
right, (b) recovers it. A second check does the same with a body feature: num_videos > 0 from slug words.
Output: results/tables/k6_alignment_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kwlib import FIRST_SHIFTED, TAB, frame  # noqa: E402
from newspop.stories import channel, slug_text  # noqa: E402

rows = []
T = frame("true")
early = (T.publish_date < FIRST_SHIFTED).to_numpy()
fit = early & (T.publish_date < "2013-05-01").to_numpy(); held = early & ~fit
late = ~early & T.url.notna().to_numpy()
slug = {"as distributed": slug_text(T.url_as_given).fillna(""), "re-aligned": slug_text(T.url.fillna("")).fillna("")}
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(slug["re-aligned"][fit])
TARGETS = {"section (7 classes), accuracy": channel(T), "has video (num_videos > 0), AUC": (T.num_videos.to_numpy() > 0).astype(int)}
for tname, t in TARGETS.items():
    clf = LogisticRegression(C=10, max_iter=2000).fit(vec.transform(slug["re-aligned"][fit]), t[fit])

    def score(text, m):
        X = vec.transform(text[m])
        if "AUC" in tname:
            return roc_auc_score(t[m], clf.predict_proba(X)[:, 1])
        return float(np.mean(clf.predict(X) == t[m]))

    base = max(pd.Series(t[late]).value_counts(normalize=True)) if "accuracy" in tname else 0.5
    rows.append(dict(target=tname, rows="2013-05-01..2013-06-05 (held out, file aligned)", url="as distributed", score=score(slug["as distributed"], held), n=int(held.sum())))
    for q, m in {"2013-06-06..2013-12-31": late & (T.publish_date < "2014-01-01").to_numpy(), "2014": late & (T.publish_date >= "2014-01-01").to_numpy()}.items():
        for a in slug:
            rows.append(dict(target=tname, rows=q, url=a, score=score(slug[a], m), n=int(m.sum())))
    rows.append(dict(target=tname, rows="chance (majority class / 0.5)", url="", score=base, n=int(late.sum())))
R = pd.DataFrame(rows); R.to_csv(TAB / "k6_alignment_check.csv", index=False)
print(R.round(3).to_string(index=False))
