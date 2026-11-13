"""Jev API: same-day, same-section Mashable pairs; which was shared more? Both option orders. usage: N_PAIRS"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
from newspop import stories as st
import jevapi
n = int(sys.argv[1])
df = load().reset_index(drop=True)
df["title"] = st.slug_text(df.url).fillna("")
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
df["day"] = pd.to_datetime(df.publish_date).dt.date
P0 = pd.read_csv("/home/user/jev/mash_pairs.csv")          # the CPU pilot pairs come first
rng = np.random.default_rng(7); extra = []
groups = [g.index.to_numpy() for _, g in df[df.title.str.len() > 0].groupby(["day", "channel"]) if len(g) >= 2]
while len(P0) + len(extra) < n:
    g = groups[rng.integers(len(groups))]; a, b = rng.choice(g, 2, replace=False)
    if df.shares[a] != df.shares[b]:
        extra.append((a, b))
P = pd.concat([P0, pd.DataFrame(extra, columns=["a", "b"])]).drop_duplicates().reset_index(drop=True).iloc[:n]
Q = "Which of these two articles was shared more often on social media?"


def job(i, a, b, order):
    d = pd.Timestamp(df.publish_date[i])
    state = (f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[i]} section. "
             "Only their headlines (taken from the URL) are shown.")
    x, y = (df.title[a], df.title[b]) if order == 0 else (df.title[b], df.title[a])
    return state, {"more_shared": {"type": "choice", "instructions": Q, "criteria": {"A": x, "B": y}}}


jobs = [job(a, a, b, o) for o in (0, 1) for a, b in zip(P.a, P.b)]
cache = jevapi.Cache("/home/user/jev/cache_mash_pairs.jsonl")
R = jevapi.call_many(jobs, cache=cache, workers=16)
bad = [r for r in R if isinstance(r, Exception)]
print("errors", len(bad), bad[:2])
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["more_shared"]["probabilities"]["A"] for r in R])
P["p0"], P["p1"] = pA[:len(P)], pA[len(P):]
P.to_csv("/home/user/jev/api_mash_pairs.csv", index=False)
print("done", len(P))
