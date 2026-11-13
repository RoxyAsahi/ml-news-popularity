"""Contamination check: winning headlines were published on upworthy.com; losers only in the archive. Jev accuracy on
significant pairs where neither headline is the test's winner (first_place), vs pairs that include the winner."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ml-news-popularity/src"); sys.path.insert(0, "/home/user/jev")
from newspop import upw_rank as ur
import jevapi
S = ur.setup("confirmatory"); H = S.H.to_numpy()[S.hid]
first = S.d.first_place.fillna(False).astype(bool).to_numpy()
sig = np.where(np.abs(S.zz) > 1.96)[0]
nw = sig[~first[S.ev_a[sig]] & ~first[S.ev_b[sig]]]
ww = sig[first[S.ev_a[sig]] | first[S.ev_b[sig]]]
rng = np.random.default_rng(9)
sets = {"no_winner": rng.choice(nw, min(2000, len(nw)), replace=False), "with_winner": rng.choice(ww, 2000, replace=False)}
STATE = ("Upworthy, a viral news website, ran a randomized A/B test of headlines for the same story. "
         "Each headline was shown to a random share of the site's visitors.")
Q = "Which headline got the higher click-through rate?"
sc = np.load("/home/user/explore/oof_score_confirmatory_roberta.npy")
print("sig pairs", len(sig), "no-winner", len(nw), "with-winner", len(ww))
for nm, pick in sets.items():
    jobs = [(STATE, {"winner": {"type": "choice", "instructions": Q, "criteria": {"A": H[(S.ev_a if o == 0 else S.ev_b)[i]], "B": H[(S.ev_b if o == 0 else S.ev_a)[i]]}}})
            for o in (0, 1) for i in pick]
    R = jevapi.call_many(jobs, cache=jevapi.Cache("/home/user/jev/cache_upw_pairs.jsonl"), workers=24, log_every=10**9)
    pA = np.array([r["answers"]["winner"]["probabilities"]["A"] for r in R]); n = len(pick)
    p = (pA[:n] + 1 - pA[n:]) / 2; zz = S.zz[pick]
    acc = np.mean((p > .5) == (zz > 0)); se = np.sqrt(acc * (1 - acc) / n)
    dr = sc[S.ev_a[pick]] - sc[S.ev_b[pick]]
    print(f"{nm}: n={n} Jev {acc:.3f} ± {1.96*se:.3f}   RoBERTa (supervised) {np.mean((dr > 0) == (zz > 0)):.3f}   median |z| {np.median(np.abs(zz)):.2f}")
