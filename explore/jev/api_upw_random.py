"""Jev on a random (not significance-filtered) sample of Upworthy within-cell pairs, both orders, for the transfer law."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
import jevapi
n = int(sys.argv[1])
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]; d = S.d
pick = np.random.default_rng(5).choice(len(S.ev_a), n, replace=False)
STATE = ("Upworthy, a viral news website, ran a randomized A/B test of headlines for the same story. "
         "Each headline was shown to a random share of the site's visitors.")
Q = "Which headline got the higher click-through rate?"
jobs = [(STATE, {"winner": {"type": "choice", "instructions": Q, "criteria": {"A": H[(S.ev_a if o == 0 else S.ev_b)[i]], "B": H[(S.ev_b if o == 0 else S.ev_a)[i]]}}})
        for o in (0, 1) for i in pick]
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_upw_pairs.jsonl"), workers=24, log_every=2000)
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["winner"]["probabilities"]["A"] for r in R])
clk, imp = d.clicks.to_numpy(float), d.impressions.to_numpy(float)
lg = lambda c, m: np.log((c + .5) / (m - c + .5))
a, b = S.ev_a[pick], S.ev_b[pick]
D = pd.DataFrame(dict(pair=pick, p0=pA[:n], p1=pA[n:], obs=lg(clk[a], imp[a]) - lg(clk[b], imp[b]),
                      nv=1 / (clk[a] + .5) + 1 / (imp[a] - clk[a] + .5) + 1 / (clk[b] + .5) + 1 / (imp[b] - clk[b] + .5), zz=S.zz[pick]))
D.to_csv("/home/user/jev/api_upw_random.csv", index=False); print("errors", int(np.isnan(pA).sum()))
