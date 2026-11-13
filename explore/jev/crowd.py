"""(A) Same-story crowding and first-mover effects from real headlines, using only earlier days (ex ante);
(B) do text signals help more on new (cold-start) topics?"""
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
sys.path.insert(0, "/home/user/ml-news-popularity/src")
from newspop.data import load
df = load().reset_index(drop=True); y = np.log1p(df.shares.to_numpy()); n = len(df)
T = pd.read_csv("/home/user/explore/mash_titles.csv").title.fillna("")
day = pd.to_datetime(df.publish_date).dt.normalize(); dn = ((day - day.min()).dt.days).to_numpy()
V = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(T)
o = np.argsort(dn, kind="stable")
feat = {k: np.zeros(n) for k in ["prev1_30", "prev3_30", "prev7_30", "prev7_50", "maxsim7", "maxsim30", "same_30", "rank_in_story"]}
for i in range(n):
    lo = np.searchsorted(dn[o], dn[i] - 30); hi = np.searchsorted(dn[o], dn[i] + 1)
    cand = o[lo:hi]; cand = cand[cand != i]
    if len(cand) == 0:
        continue
    s = (V[cand] @ V[i].T).toarray().ravel(); gap = dn[i] - dn[cand]
    prev = gap > 0
    feat["maxsim30"][i] = s[prev].max() if prev.any() else 0
    p7 = prev & (gap <= 7)
    feat["maxsim7"][i] = s[p7].max() if p7.any() else 0
    feat["prev1_30"][i] = np.sum((s >= .3) & prev & (gap <= 1)); feat["prev3_30"][i] = np.sum((s >= .3) & prev & (gap <= 3))
    feat["prev7_30"][i] = np.sum((s >= .3) & p7); feat["prev7_50"][i] = np.sum((s >= .5) & p7)
    feat["same_30"][i] = np.sum((s >= .3) & (gap == 0))
F = pd.DataFrame(feat); F.to_csv("/home/user/explore/crowd_feats.csv", index=False)
O = pd.read_csv("/home/user/explore/mash_oof.csv"); res = y - O.hgb.to_numpy()
print("share of articles with a similar (>=0.3) article in the previous 7 days: %.3f" % np.mean(F.prev7_30 > 0))
for k in ["prev7_30", "prev7_50", "same_30"]:
    g = pd.Series(res).groupby(np.minimum(F[k], 3)).agg(["mean", "count"])
    print(k, "residual log-shares (vs HGB) by count 0/1/2/3+:", {int(i): (round(r["mean"], 3), int(r["count"])) for i, r in g.iterrows()})
# prediction: HGB with vs without crowding features (same folds as baseline)
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
pop = (df.shares >= 1400).astype(int)
for name, XX in [("HGB base", X), ("HGB + ex-ante crowding", pd.concat([X, F.drop(columns=["same_30", "rank_in_story"])], axis=1)),
                 ("HGB + crowding incl. same-day", pd.concat([X, F.drop(columns=["rank_in_story"])], axis=1))]:
    p = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(XX):
        p[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(XX.iloc[tr], y[tr]).predict(XX.iloc[te])
    np.save(f"/home/user/explore/oof_{name.replace(' ', '_').replace('+', 'p')}.npy", p)
    print(f"{name}: AUC {roc_auc_score(pop, p):.4f}  corr {np.corrcoef(p, y)[0, 1]:.4f}")
