"""Paraphrase test: 250 same-day same-section Mashable pairs from January-May 2013 (where the judge has signal), with
every headline reworded (same meaning, names and numbers; different wording; written by an AI assistant, para_out.txt).
Same prompt as api_mash_titles.py, both orders. If the judge's signal were verbatim recall it should vanish."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop.data import load
import jevapi
df = load().reset_index(drop=True)
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
para = {int(k): v for k, v in (l.rstrip("\n").split("|", 1) for l in open("/home/user/jev/para_out.txt") if l.strip())}
PP = pd.read_csv("/home/user/jev/para_pairs.csv"); M = pd.read_csv("/home/user/jev/api_mash_titles_section.csv")
Q = "Which of these two articles was shared more often on social media?"
jobs = []
for o in (0, 1):
    for a, b in zip(PP.a, PP.b):
        d = pd.Timestamp(df.publish_date[a]); x, y_ = (a, b) if o == 0 else (b, a)
        state = f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[a]} section."
        jobs.append((state, {"more_shared": {"type": "choice", "instructions": Q, "criteria": {"A": para[x], "B": para[y_]}}}))
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_paraphrase.jsonl"), workers=24, log_every=10**9)
pA = np.array([np.nan if isinstance(r, Exception) else r["answers"]["more_shared"]["probabilities"]["A"] for r in R]); n = len(PP)
PP["p0_para"], PP["p1_para"] = pA[:n], pA[n:]
PP["p0_orig"], PP["p1_orig"] = M.p0_title.to_numpy()[PP.k], M.p1_title.to_numpy()[PP.k]
PP.to_csv("/home/user/jev/api_paraphrase.csv", index=False)
y = np.log1p(df.shares.to_numpy()); t = y[PP.a] > y[PP.b]
po, pp = (PP.p0_orig + 1 - PP.p1_orig) / 2, (PP.p0_para + 1 - PP.p1_para) / 2
rng = np.random.default_rng(0)
def ci(c): b = [c[rng.integers(0, n, n)].mean() for _ in range(2000)]; return c.mean().round(3), np.quantile(b, [.025, .975]).round(3)
print("errors", int(np.isnan(pA).sum()))
print("original", ci(((po > .5) == t).to_numpy().astype(float))); print("paraphrased", ci(((pp > .5) == t).to_numpy().astype(float)))
print("same decision", np.mean((po > .5) == (pp > .5)).round(3), "corr of probabilities", np.corrcoef(po, pp)[0, 1].round(3))
d = ((pp > .5) == t).astype(float) - ((po > .5) == t).astype(float); print("paired diff", ci(d.to_numpy()))
