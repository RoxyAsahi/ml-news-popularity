"""Repaired data: 400 random Mashable section pairs from the whole period (k = row of rep_mash_titles_section.csv).
Open-weights baseline: JEV-9B (int8, CPU) on the same prompts as the API runs. 400 Upworthy significant pairs and
400 Mashable same-day same-section pairs with real headlines, both option orders. Resumable."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
from newspop.data import load
from newspop.titles import aligned_titles
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]
U = pd.read_csv("/home/user/jev/api_upw_pairs.csv").iloc[:0]
USTATE = ("Upworthy, a viral news website, ran a randomized A/B test of headlines for the same story. "
          "Each headline was shown to a random share of the site's visitors.")
UQ = "Which headline got the higher click-through rate?"
df = load().reset_index(drop=True); df["title"] = aligned_titles(df)
NAMES = dict(lifestyle="Lifestyle", entertainment="Entertainment", bus="Business", socmed="Social Media", tech="Tech", world="World")
df["channel"] = "Other"
for c, nm in NAMES.items():
    df.loc[df[f"data_channel_is_{c}"] == 1, "channel"] = nm
M = pd.read_csv("/home/user/jev/rep_mash_titles_section.csv")
late = np.sort(np.random.default_rng(1).choice(len(M), N, replace=False))
MQ = "Which of these two articles was shared more often on social media?"
items = []
for k, i in enumerate(U.pair):
    for o in (0, 1):
        x, y = (H[S.ev_a[i]], H[S.ev_b[i]]) if o == 0 else (H[S.ev_b[i]], H[S.ev_a[i]])
        items.append(("upw", k, o, USTATE, UQ, [x, y]))
for k in late:
    a, b = M.a[k], M.b[k]
    d = pd.Timestamp(df.publish_date[a])
    st = f"Two articles published on Mashable on {d:%A, %d %B %Y}, both in the {df.channel[a]} section."
    for o in (0, 1):
        x, y = (df.title[a], df.title[b]) if o == 0 else (df.title[b], df.title[a])
        items.append(("mash", k, o, st, MQ, [x, y]))
out = "/home/user/jev/rep_local_runs.csv"
done = set()
if os.path.exists(out):
    D = pd.read_csv(out); done = set(zip(D.set, D.k, D.order))
items = [x for x in items if (x[0], x[1], x[2]) not in done]
items.sort(key=lambda x: len(x[3]) + sum(len(s) for s in x[5]))
import jev
model = jev.load(); t0 = time.time()
for s in range(0, len(items), 8):
    ch = items[s:s + 8]
    pr = jev.decide(model, "choice", [(x[3], x[4], x[5]) for x in ch])
    pd.DataFrame([dict(set=x[0], k=x[1], order=x[2], p_first=p[0]) for x, p in zip(ch, pr)]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print(s + len(ch), len(items), f"{time.time() - t0:.0f}s", flush=True)
