"""Evaluate pairwise popularity judgments: file with a, b, p0 (P(a) when a listed first), p1 (P(first) when b first)."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
P = pd.read_csv(sys.argv[1]); O = pd.read_csv("/home/user/explore/mash_oof.csv")
P["p"] = (P.p0 + (1 - P.p1)) / 2
P["truth"] = (O.y[P.a].to_numpy() > O.y[P.b].to_numpy()).astype(int)
P["d_hgb"] = O.hgb[P.a].to_numpy() - O.hgb[P.b].to_numpy()
P["gap"] = np.abs(O.y[P.a].to_numpy() - O.y[P.b].to_numpy())
acc = lambda s, t: float(np.mean((s > 0) == (t == 1)))
print(f"n={len(P)}  mean P(first)={np.r_[P.p0, P.p1].mean():.3f}  order-consistent={np.mean((P.p0 > .5) == (P.p1 < .5)):.3f}")
print(f"Jev acc {acc(P.p - .5, P.truth):.3f} AUC {roc_auc_score(P.truth, P.p):.3f} | HGB acc {acc(P.d_hgb, P.truth):.3f} AUC {roc_auc_score(P.truth, P.d_hgb):.3f}")
for q in [1, .5, .25, .1]:
    k = max(1, int(q * len(P)))
    oj = np.argsort(-np.abs(P.p - .5).to_numpy())[:k]; oh = np.argsort(-np.abs(P.d_hgb).to_numpy())[:k]
    print(f"  most-confident {q:>4}: Jev {acc(P.p.iloc[oj] - .5, P.truth.iloc[oj]):.3f}  HGB {acc(P.d_hgb.iloc[oh], P.truth.iloc[oh]):.3f}")
lg = np.log(np.clip(P.p, 1e-4, 1 - 1e-4) / (1 - np.clip(P.p, 1e-4, 1 - 1e-4)))
X = np.c_[P.d_hgb, lg]
for name, cols in [("HGB", [0]), ("Jev", [1]), ("HGB+Jev", [0, 1])]:
    p = cross_val_predict(LogisticRegression(), X[:, cols], P.truth, cv=10, method="predict_proba")[:, 1]
    print(f"  {name} 10-fold AUC {roc_auc_score(P.truth, p):.3f}")
big = P.gap > np.log(2)
print(f"  pairs with >=2x share gap: {big.sum()}  Jev {acc(P.p[big] - .5, P.truth[big]):.3f}  HGB {acc(P.d_hgb[big], P.truth[big]):.3f}")
print(f"  corr(Jev logit, HGB diff) {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.3f}")
