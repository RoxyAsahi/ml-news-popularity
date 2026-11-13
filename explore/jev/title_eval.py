import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
O = pd.read_csv("/home/user/explore/mash_oof.csv")
for mode in ["section", "day"]:
    P = pd.read_csv(f"/home/user/jev/api_mash_titles_{mode}.csv")
    t = (O.y[P.a].to_numpy() > O.y[P.b].to_numpy()).astype(int)
    dh = O.hgb[P.a].to_numpy() - O.hgb[P.b].to_numpy()
    acc = lambda s, m=None: float(np.mean((s > 0) == (t == 1)))
    print(f"== {mode} pairs n={len(P)}  HGB acc {acc(dh):.3f} AUC {roc_auc_score(t, dh):.3f}")
    L = {}
    for f in ["title", "slugt"]:
        p = (P[f"p0_{f}"] + 1 - P[f"p1_{f}"]) / 2; L[f] = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        cons = np.mean((P[f"p0_{f}"] > .5) == (P[f"p1_{f}"] < .5))
        conf = np.abs(p - .5).to_numpy(); o = np.argsort(-conf)
        sel = {q: round(float(np.mean(((p.to_numpy() > .5) == (t == 1))[o[: int(q * len(P))]])), 3) for q in [.5, .25, .1]}
        print(f"  Jev[{f}] acc {acc(p - .5):.3f} AUC {roc_auc_score(t, p):.3f} order-consistent {cons:.2f}  top-confidence acc {sel}")
    X = np.c_[dh, L["title"], L["slugt"]]
    for nm, cols in [("HGB", [0]), ("Jev title", [1]), ("HGB+Jev title", [0, 1]), ("HGB+Jev slug", [0, 2])]:
        pp = cross_val_predict(LogisticRegression(), X[:, cols], t, cv=10, method="predict_proba")[:, 1]
        print(f"  {nm:15s} 10-fold acc {np.mean((pp > .5) == t):.3f} AUC {roc_auc_score(t, pp):.3f}")
    # cascade: where HGB unsure (bottom half of |dh|), how does Jev do?
    lo = np.abs(dh) < np.median(np.abs(dh))
    pt = (P.p0_title + 1 - P.p1_title) / 2
    print(f"  HGB-unsure half: HGB acc {np.mean((dh[lo] > 0) == (t[lo] == 1)):.3f}  Jev title acc {np.mean((pt[lo] > .5) == (t[lo] == 1)):.3f}")
