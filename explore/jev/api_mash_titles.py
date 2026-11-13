"""Jev on Mashable with the real headlines (sitemap) vs URL keywords, same-day pairs. usage: N_PAIRS MODE (section|day)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
from newspop import stories as st
import jevapi
n, mode = int(sys.argv[1]), sys.argv[2]
df = load().reset_index(drop=True)
df["slugt"] = st.slug_text(df.url).fillna("")
df["title"] = pd.read_csv("/home/user/explore/mash_titles.csv").title
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
df["day"] = pd.to_datetime(df.publish_date).dt.date
ok = df.title.notna() & (df.slugt.str.len() > 0)
keys = ["day", "channel"] if mode == "section" else ["day"]
groups = [g.index.to_numpy() for _, g in df[ok].groupby(keys) if len(g) >= 2]
rng = np.random.default_rng(21 if mode == "section" else 22); P = set()
while len(P) < n:
    g = groups[rng.integers(len(groups))]; a, b = rng.choice(g, 2, replace=False)
    if df.shares[a] != df.shares[b]:
        P.add((min(a, b), max(a, b)))
P = pd.DataFrame(sorted(P), columns=["a", "b"])
Q = "Which of these two articles was shared more often on social media?"


def job(a, b, o, field):
    d = pd.Timestamp(df.publish_date[a])
    x, y = (a, b) if o == 0 else (b, a)
    if mode == "section":
        state = f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[a]} section."
        cx, cy = df[field][x], df[field][y]
    else:
        state = f"Two articles published on Mashable on {d:%A, %d %B %Y}."
        cx, cy = f"{df[field][x]} (section: {df.channel[x]})", f"{df[field][y]} (section: {df.channel[y]})"
    if field == "slugt":
        state += " Only keywords from each article's URL are shown."
    return state, {"more_shared": {"type": "choice", "instructions": Q, "criteria": {"A": cx, "B": cy}}}


cache = jevapi.Cache(f"/home/user/jev/cache_mash_titles.jsonl")
for field in ["title", "slugt"]:
    R = jevapi.call_many([job(a, b, o, field) for o in (0, 1) for a, b in zip(P.a, P.b)], cache=cache, workers=24, log_every=2000)
    pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["more_shared"]["probabilities"]["A"] for r in R])
    print(field, "errors", int(np.isnan(pA).sum()))
    P[f"p0_{field}"], P[f"p1_{field}"] = pA[:n], pA[n:]
P.to_csv(f"/home/user/jev/api_mash_titles_{mode}.csv", index=False)
