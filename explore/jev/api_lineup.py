"""Jev sees the whole day's lineup of Mashable headlines (one Choice over all of the day's articles, 3 random orders)
and says which was shared most. Output: per-article probability (mean over orders)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
import jevapi
df = load().reset_index(drop=True); df["title"] = pd.read_csv("/home/user/explore/mash_titles.csv").title
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
df["day"] = pd.to_datetime(df.publish_date).dt.date
ok = df.title.notna()
jobs, meta = [], []
rng = np.random.default_rng(0)
for day, g in df[ok].groupby("day"):
    if len(g) < 5 or len(g) > 250:
        continue
    for rep in range(3):
        ix = rng.permutation(g.index.to_numpy())
        crit = {f"a{j}": f"{df.title[i]} (section: {df.channel[i]})" for j, i in enumerate(ix)}
        state = f"All articles Mashable published on {pd.Timestamp(day):%A, %d %B %Y}."
        jobs.append((state, {"top": {"type": "choice", "instructions": "Which of these articles was shared the most on social media?", "criteria": crit}}))
        meta.append(ix)
print(len(jobs), "requests")
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_lineup.jsonl"), workers=16, log_every=500)
acc = {}
for ix, r in zip(meta, R):
    if isinstance(r, Exception):
        print("err", r); continue
    pr = r["answers"]["top"]["probabilities"]
    for j, i in enumerate(ix):
        acc.setdefault(i, []).append(pr.get(f"a{j}", np.nan))
out = pd.DataFrame({"row": list(acc), "p_top": [np.nanmean(v) for v in acc.values()]})
out.to_csv("/home/user/jev/api_lineup.csv", index=False); print(len(out))
