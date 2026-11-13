"""(Repaired data.) Jev given the headline plus a few of the dataset's own features for each article (same-day same-section pairs)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
from newspop.titles import aligned_titles
import jevapi
df = load().reset_index(drop=True); df["title"] = aligned_titles(df)
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
P = pd.read_csv("/home/user/jev/rep_mash_titles_section.csv").sample(2000, random_state=0).sort_index().reset_index(drop=True)


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
P.to_csv("/home/user/jev/rep_feat_pairs.csv", index=False)
