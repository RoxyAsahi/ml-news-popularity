"""Which headline words go with more or fewer shares? (repaired data, descriptive)

Ridge regression of log shares, centred within section and month, on binary indicators of headline words that
occur in at least 40 headlines; the 15 largest positive and negative coefficients, with how often the word occurs
and the median shares of the articles that carry it. Descriptive only: no claim about causes.
Output: results/tables/v6_headline_terms.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402
from newspop.stories import channel  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

df = load().reset_index(drop=True); title = aligned_titles(df); ok = title.notna().to_numpy()
df, title = df[ok].reset_index(drop=True), title[ok].reset_index(drop=True)
y = np.log1p(df.shares.to_numpy())
key = pd.Series(channel(df)) + pd.to_datetime(df.publish_date).dt.to_period("M").astype(str)
r = y - pd.Series(y).groupby(key).transform("mean").to_numpy()
v = CountVectorizer(min_df=40, binary=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
Xw = v.fit_transform(title); m = Ridge(alpha=30).fit(Xw, r)
words = np.array(v.get_feature_names_out()); n = np.asarray(Xw.sum(0)).ravel()
med = np.array([np.median(df.shares.to_numpy()[Xw[:, j].toarray().ravel() > 0]) for j in range(Xw.shape[1])])
R = pd.DataFrame({"word": words, "coef": m.coef_, "headlines": n, "median_shares": med}).sort_values("coef")
out = pd.concat([R.tail(15)[::-1].assign(side="more shares"), R.head(15).assign(side="fewer shares")])
out.to_csv(ROOT / "results" / "tables" / "v6_headline_terms.csv", index=False)
print(out.round(3).to_string(index=False)); print("vocabulary", len(words), "articles", len(df))
