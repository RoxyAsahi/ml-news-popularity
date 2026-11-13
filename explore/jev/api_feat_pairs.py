"""Jev given the headline plus a few of the dataset's own features for each article (same-day same-section pairs)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
import jevapi
df = load().reset_index(drop=True); df["title"] = pd.read_csv("/home/user/explore/mash_titles.csv").title
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
P = pd.read_csv("/home/user/jev/api_mash_titles_section.csv").iloc[:2000].copy()


def desc(i):
    r = df.loc[i]
    return (f"{r.title} | {int(r.n_tokens_content)} words, {int(r.num_imgs)} images, {int(r.num_videos)} videos, "
            f"{int(r.num_hrefs)} links | earlier Mashable articles with the same keywords got {int(r.kw_avg_avg):,} shares on average | "
            f"Mashable articles it links to got {int(r.self_reference_avg_sharess):,} shares on average")


Q = "Which of these two articles was shared more often on social media?"
jobs = []
for o in (0, 1):
    for a, b in zip(P.a, P.b):
        d = pd.Timestamp(df.publish_date[a]); x, y = (a, b) if o == 0 else (b, a)
        jobs.append((f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[a]} section.",
                     {"more_shared": {"type": "choice", "instructions": Q, "criteria": {"A": desc(x), "B": desc(y)}}}))
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_feat.jsonl"), workers=24, log_every=10**9)
pA = np.array([r["answers"]["more_shared"]["probabilities"]["A"] for r in R]); n = len(P)
P["p0_feat"], P["p1_feat"] = pA[:n], pA[n:]
P.to_csv("/home/user/jev/api_feat_pairs.csv", index=False)
O = pd.read_csv("/home/user/explore/mash_oof.csv"); t = O.y[P.a].to_numpy() > O.y[P.b].to_numpy()
p = (P.p0_feat + 1 - P.p1_feat) / 2; dh = O.hgb[P.a].to_numpy() - O.hgb[P.b].to_numpy()
kw = df.kw_avg_avg[P.a].to_numpy() - df.kw_avg_avg[P.b].to_numpy()
print("Jev(title+features) %.3f  HGB %.3f  kw_avg_avg alone %.3f  order-consistent %.2f" % (np.mean((p > .5) == t), np.mean((dh > 0) == t), np.mean((kw > 0) == t), np.mean((P.p0_feat > .5) == (P.p1_feat < .5))))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lg = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for nm, X in [("HGB", np.c_[dh]), ("HGB+Jev(feat)", np.c_[dh, lg])]:
    print(nm, "10-fold acc %.3f" % cross_val_score(LogisticRegression(), X, t, cv=10).mean())
