"""Out-of-fold baselines for Mashable pairwise popularity judgments (same day, same section): HGB on the 58 features,
TF-IDF ridge on the URL headline. Saves explore/mash_oof.csv and reports accuracy on all within-day-section pairs."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sys.path.insert(0, "/home/user/ml-news-popularity/src")
from newspop.data import load
from newspop import stories as st
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy())
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
T = st.slug_text(df.url).fillna("")
oof = pd.DataFrame(index=df.index, columns=["hgb", "tfidf"], dtype=float)
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    oof.loc[te, "hgb"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(T[tr])
    oof.loc[te, "tfidf"] = Ridge(alpha=10).fit(v.transform(T[tr]), y[tr]).predict(v.transform(T[te]))
oof["y"] = y; oof.to_csv("/home/user/explore/mash_oof.csv", index=False)
P = pd.read_csv("/home/user/jev/mash_pairs.csv")
for m in ["hgb", "tfidf"]:
    print(m, "acc on JEV pairs", np.mean(np.sign(oof[m][P.a].to_numpy() - oof[m][P.b].to_numpy()) == np.sign(y[P.a] - y[P.b])).round(4))
