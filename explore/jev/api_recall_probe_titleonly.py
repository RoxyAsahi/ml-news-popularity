"""Variant of api_recall_probe.py: headline only (no URL, date or section). Memorization probe: shown one Mashable article (URL, date, section, headline), can the judge say how often it was
shared? 1,000 random articles from 2013 Q1-Q2 and 1,000 from 2014. Choice over four share bands; the expected band
index is compared with the true shares (Spearman, AUC for popular). A model that remembers articles should do better
on the period where its pairwise accuracy is higher."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
import jevapi
df = load().reset_index(drop=True); df["title"] = pd.read_csv("/home/user/explore/mash_titles.csv").title
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
d = pd.to_datetime(df.publish_date); ok = df.title.notna()
rng = np.random.default_rng(5)
early = rng.choice(np.flatnonzero(ok & (d < "2013-07-01")), 1000, replace=False)
late = rng.choice(np.flatnonzero(ok & (d >= "2014-01-01")), 1000, replace=False)
BANDS = {"A": "fewer than 1,000 shares", "B": "1,000 to 2,000 shares", "C": "2,000 to 5,000 shares", "D": "more than 5,000 shares"}
Q = "About how many times was this article shared on social media (Facebook, Twitter, Google+, LinkedIn, etc.)?"
jobs = []
for i in np.r_[early, late]:
    state = f"A Mashable article. Headline: {df.title[i]}"
    jobs.append((state, {"shares": {"type": "choice", "instructions": Q, "criteria": BANDS}}))
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_recall_titleonly.jsonl"), workers=24, log_every=10**9)
P = np.array([[np.nan] * 4 if isinstance(r, Exception) else [r["answers"]["shares"]["probabilities"][k] for k in "ABCD"] for r in R])
out = pd.DataFrame(dict(row=np.r_[early, late], period=["2013H1"] * 1000 + ["2014"] * 1000, pA=P[:, 0], pB=P[:, 1], pC=P[:, 2], pD=P[:, 3]))
out.to_csv("/home/user/jev/api_recall_probe_titleonly.csv", index=False)
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
out["exp_band"] = P @ np.arange(4); out["shares"] = df.shares.to_numpy()[out.row]
for per, g in out.groupby("period"):
    print(per, "spearman", round(spearmanr(g.exp_band, g.shares)[0], 3), "AUC popular", round(roc_auc_score(g.shares >= 1400, g.exp_band), 3))
