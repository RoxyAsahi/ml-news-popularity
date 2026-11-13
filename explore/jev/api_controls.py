"""Two controls. (1) Upworthy headlines degraded to Mashable-style slugs (lowercase, no punctuation or stopwords,
truncated to a slug-like length): is Jev's skill on Upworthy lost when headlines look like slugs?
(2) Mashable pairs from the same day but any section: can Jev see topic-level appeal?"""
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
from newspop.data import load
from newspop import stories as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
import jevapi
df = load().reset_index(drop=True); df["title"] = st.slug_text(df.url).fillna("")
L = df.title.str.split().str.len(); Ls = L[L > 0].to_numpy()
print("Mashable slug words: mean", Ls.mean().round(2), "median", np.median(Ls))
rng = np.random.default_rng(3)


def slugify(h):
    w = [t for t in re.sub(r"[^a-z0-9 ]", " ", h.lower()).split() if t not in SW]
    return " ".join(w[: int(rng.choice(Ls))])


# (1)
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]
D = pd.read_csv("/home/user/jev/api_upw_pairs.csv")
STATE = ("Upworthy, a viral news website, ran a randomized A/B test of headlines for the same story. "
         "Each headline was shown to a random share of the site's visitors. Headlines are shown in URL-slug form.")
Q = "Which headline got the higher click-through rate?"
sa = {i: slugify(H[S.ev_a[i]]) for i in D.pair}; sb = {i: slugify(H[S.ev_b[i]]) for i in D.pair}
keep = [i for i in D.pair if sa[i] != sb[i] and sa[i] and sb[i]]
jobs = [(STATE, {"winner": {"type": "choice", "instructions": Q, "criteria": {"A": (sa if o == 0 else sb)[i], "B": (sb if o == 0 else sa)[i]}}})
        for o in (0, 1) for i in keep]
R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_controls.jsonl"))
pA = np.array([r["answers"]["winner"]["probabilities"]["A"] for r in R]); n = len(keep)
p = (pA[:n] + 1 - pA[n:]) / 2; zz = S.zz[keep]
Dk = D.set_index("pair").loc[keep]; pfull = (Dk.p0 + 1 - Dk.p1) / 2
print(f"(1) Upworthy, slug form: n={n} acc {np.mean((p > .5) == (zz > 0)):.3f}  vs full headlines on same pairs {np.mean((pfull > .5) == (zz > 0)):.3f}")
pd.DataFrame(dict(pair=keep, p_slug=p)).to_csv("/home/user/jev/api_upw_slug.csv", index=False)
# (2)
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
df["day"] = pd.to_datetime(df.publish_date).dt.date
groups = [g.index.to_numpy() for _, g in df[df.title.str.len() > 0].groupby("day") if len(g) >= 2]
P = []
while len(P) < 1500:
    g = groups[rng.integers(len(groups))]; a, b = rng.choice(g, 2, replace=False)
    if df.shares[a] != df.shares[b]:
        P.append((a, b))
P = pd.DataFrame(P, columns=["a", "b"])
Q2 = "Which of these two articles was shared more often on social media?"


def job(a, b, o):
    d = pd.Timestamp(df.publish_date[a])
    x, y = (a, b) if o == 0 else (b, a)
    return (f"Two articles published on Mashable on {d:%A, %d %B %Y}. Only their headlines (taken from the URL) and sections are shown.",
            {"more_shared": {"type": "choice", "instructions": Q2, "criteria": {"A": f"{df.title[x]} (section: {df.channel[x]})",
                                                                               "B": f"{df.title[y]} (section: {df.channel[y]})"}}})


R = jevapi.call_many([job(a, b, o) for o in (0, 1) for a, b in zip(P.a, P.b)], cache=jevapi.Cache("/home/user/jev/cache_controls.jsonl"))
pA = np.array([r["answers"]["more_shared"]["probabilities"]["A"] for r in R])
P["p0"], P["p1"] = pA[:len(P)], pA[len(P):]
P.to_csv("/home/user/jev/api_mash_dayonly.csv", index=False)
