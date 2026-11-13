"""Second question wording (robustness): 1,000 Mashable same-day same-section pairs and 1,000 Upworthy significant
within-cell pairs, the same pairs as the main runs, both orders, with a different state line and question."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
from newspop.data import load
import jevapi
N = 1000
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]
U = pd.read_csv("/home/user/jev/api_upw_pairs.csv").iloc[:N]
df = load().reset_index(drop=True); df["title"] = pd.read_csv("/home/user/explore/mash_titles.csv").title
M = pd.read_csv("/home/user/jev/api_mash_titles_section.csv").iloc[:N]
US, UQ = "Two headlines written for the same story on Upworthy.", "Which headline would make more readers click?"
MS = "Two news articles from Mashable, a site about technology, digital culture and entertainment."
MQ = "Which article did readers share more on Facebook, Twitter, Google+ and LinkedIn?"
jobs = []
for i in U.pair:
    for o in (0, 1):
        x, y = (H[S.ev_a[i]], H[S.ev_b[i]]) if o == 0 else (H[S.ev_b[i]], H[S.ev_a[i]])
        jobs.append((US, {"q": {"type": "choice", "instructions": UQ, "criteria": {"A": x, "B": y}}}))
for a, b in zip(M.a, M.b):
    for o in (0, 1):
        x, y = (df.title[a], df.title[b]) if o == 0 else (df.title[b], df.title[a])
        jobs.append((MS, {"q": {"type": "choice", "instructions": MQ, "criteria": {"A": x, "B": y}}}))
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_prompt2.jsonl"), workers=24, log_every=10**9)
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["q"]["probabilities"]["A"] for r in R])
print("errors", int(np.isnan(pA).sum()))
pu, pm = pA[:2 * N].reshape(N, 2), pA[2 * N:].reshape(N, 2)
out = pd.concat([pd.DataFrame(dict(set="upw", k=np.arange(N), p0=pu[:, 0], p1=pu[:, 1])),
                 pd.DataFrame(dict(set="mash", k=np.arange(N), p0=pm[:, 0], p1=pm[:, 1]))])
out.to_csv("/home/user/jev/api_prompt2.csv", index=False)
y = np.log1p(df.shares.to_numpy())
for s, P, truth in [("upw", pu, U.zz.to_numpy() > 0), ("mash", pm, y[M.a] > y[M.b])]:
    p = (P[:, 0] + 1 - P[:, 1]) / 2
    print(s, "acc", np.mean((p > .5) == truth).round(4), "order-consistent", np.mean((P[:, 0] > .5) == (P[:, 1] < .5)).round(3))
