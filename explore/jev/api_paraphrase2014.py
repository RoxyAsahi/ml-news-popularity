"""Paraphrase test on later pairs: 250 same-day same-section pairs from 2014 (repaired data), drawn at random from
rep_mash_titles_section.csv, every headline reworded by an AI assistant (para2014_out.txt: same meaning, names and
numbers; different wording). Same prompt as api_mash_titles.py, both orders."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
import jevapi
df = load().reset_index(drop=True)
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
rd = lambda f: {int(k): v for k, v in (l.rstrip("\n").split("|", 1) for l in open(f) if l.strip())}
src, para = rd("/home/user/jev/para2014_src.txt"), rd("/home/user/jev/para2014_out.txt")
PP = pd.read_csv("/home/user/jev/para2014_pairs.csv"); M = pd.read_csv("/home/user/jev/rep_mash_titles_section.csv")
assert (df.url[PP.a].to_numpy() == PP.url_a).all() and (df.url[PP.b].to_numpy() == PP.url_b).all()
assert (M.url_a[PP.k].to_numpy() == PP.url_a).all() and (M.url_b[PP.k].to_numpy() == PP.url_b).all()
Q = "Which of these two articles was shared more often on social media?"
jobs = []
for o in (0, 1):
    for a, b in zip(PP.a, PP.b):
        d = pd.Timestamp(df.publish_date[a]); x, y_ = (a, b) if o == 0 else (b, a)
        state = f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[a]} section."
        jobs.append((state, {"more_shared": {"type": "choice", "instructions": Q, "criteria": {"A": para[x], "B": para[y_]}}}))
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_paraphrase2014.jsonl"), workers=24, log_every=10**9)
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["more_shared"]["probabilities"]["A"] for r in R]); n = len(PP)
PP["p0_para"], PP["p1_para"] = pA[:n], pA[n:]
PP["p0_orig"], PP["p1_orig"] = M.p0_title.to_numpy()[PP.k], M.p1_title.to_numpy()[PP.k]
for s in "ab":
    PP[f"title_{s}"] = [src[i] for i in PP[s]]; PP[f"reworded_{s}"] = [para[i] for i in PP[s]]
PP.to_csv("/home/user/jev/api_paraphrase2014.csv", index=False)
y = np.log1p(df.shares.to_numpy()); t = y[PP.a] > y[PP.b]
po, pp = (PP.p0_orig + 1 - PP.p1_orig) / 2, (PP.p0_para + 1 - PP.p1_para) / 2
rng = np.random.default_rng(0)
def ci(c): b = [c[rng.integers(0, n, n)].mean() for _ in range(2000)]; return c.mean().round(3), np.quantile(b, [.025, .975]).round(3)
tok = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))
print("errors", int(np.isnan(pA).sum()), "mean word overlap (Jaccard)", np.mean([len(tok(src[k]) & tok(para[k])) / len(tok(src[k]) | tok(para[k])) for k in src]).round(3))
print("original", ci(((po > .5) == t).to_numpy().astype(float))); print("paraphrased", ci(((pp > .5) == t).to_numpy().astype(float)))
print("same decision", np.mean((po > .5) == (pp > .5)).round(3), "corr of probabilities", np.corrcoef(po, pp)[0, 1].round(3))
d = ((pp > .5) == t).astype(float) - ((po > .5) == t).astype(float); print("paired diff", ci(d.to_numpy()))
