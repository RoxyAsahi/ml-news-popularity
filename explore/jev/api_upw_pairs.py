"""Jev API on Upworthy: significant within-cell headline pairs (confirmatory), which got the higher click rate? Both orders."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
import jevapi
n = int(sys.argv[1])
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]
sig = np.where(np.abs(S.zz) > 1.96)[0]
pick = np.random.default_rng(0).choice(sig, n, replace=False)
STATE = ("Upworthy, a viral news website, ran a randomized A/B test of headlines for the same story. "
         "Each headline was shown to a random share of the site's visitors.")
Q = "Which headline got the higher click-through rate?"
jobs = [(STATE, {"winner": {"type": "choice", "instructions": Q, "criteria": {"A": H[S.ev_a[i]] if o == 0 else H[S.ev_b[i]],
                                                                            "B": H[S.ev_b[i]] if o == 0 else H[S.ev_a[i]]}}})
        for o in (0, 1) for i in pick]
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_upw_pairs.jsonl"), workers=16)
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["winner"]["probabilities"]["A"] for r in R])
D = pd.DataFrame(dict(pair=pick, p0=pA[:n], p1=pA[n:], zz=S.zz[pick]))
D.to_csv("/home/user/jev/api_upw_pairs.csv", index=False)
D["p"] = (D.p0 + 1 - D.p1) / 2
ok = (D.p > .5) == (D.zz > 0)
print("errors", int(np.isnan(pA).sum()), "order-consistent", np.mean((D.p0 > .5) == (D.p1 < .5)).round(3), "acc", ok.mean().round(4))
sc = np.load("/home/user/explore/oof_score_confirmatory_roberta.npy")
dr = sc[S.ev_a[pick]] - sc[S.ev_b[pick]]; okr = (dr > 0) == (D.zz > 0)
print("supervised RoBERTa on same pairs", okr.mean().round(4))
for q in [1, .5, .25, .1]:
    k = int(q * n); oj = np.argsort(-np.abs(D.p - .5).to_numpy())[:k]; orr = np.argsort(-np.abs(dr))[:k]
    print(f"  top {q}: Jev {ok.iloc[oj].mean():.3f}  RoBERTa {okr.iloc[orr].mean():.3f}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lg = np.log(np.clip(D.p, 1e-4, 1 - 1e-4) / (1 - np.clip(D.p, 1e-4, 1 - 1e-4)))
y = (D.zz > 0).astype(int)
for nm, X in [("RoBERTa", np.c_[dr]), ("Jev", np.c_[lg]), ("both", np.c_[dr, lg])]:
    print(nm, "10-fold acc", cross_val_score(LogisticRegression(), X, y, cv=10).mean().round(4))
