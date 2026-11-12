"""S2 (RQ1): what is the model's skill made of? Pairwise AUC within nested contexts, past-only.

Scores: monthly rolling origin (train on all earlier months, score the month), first scored
month 2013-07. Label: shares >= 1,400. Strata: all pairs, same week, same day, same channel,
same channel x week, same channel x day, same-story twins, same-story twins in the same channel.
Two metrics: pairwise AUC on the binary label, and concordance on pairs whose shares differ by
more than 2x (|delta log shares| > log 2), which does not depend on the 1,400 threshold.
CIs: week-cluster bootstrap for context strata; story-cluster bootstrap for twins.

Outputs: results/tables/s2_anatomy.csv, data/processed/scores.csv (cache for later scripts)
"""
import numpy as np
import pandas as pd

from newspop.data import ROOT, feature_set, load
from newspop import anatomy as an, stories as st

PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
df = load()
y = (df.shares >= 1400).astype(int).to_numpy()
ls = np.log1p(df.shares.to_numpy())
week, day, chan = st.week_index(df), st.day_index(df), st.channel(df)

cache = PROC / "scores.csv"
if cache.exists():
    S = pd.read_csv(cache)
else:
    S = pd.DataFrame({"url": df.url})
    for name, model, fs in [("hgb_all", "hgb", "all"), ("logreg_all", "logreg", "all"),
                            ("hgb_noshare", "hgb", "no_kw_ref")]:
        X = df[feature_set(df, fs)].to_numpy(dtype=float)
        S[name] = an.past_only_scores(df, X, y, model=model)
        print("scored", name, flush=True)
    S.to_csv(cache, index=False)
assert (S.url.to_numpy() == df.url.to_numpy()).all()

P = st.clean_twins(df)
sid = st.story_ids(len(df), P)
P["story"] = sid[P.i]
P["same_channel"] = chan[P.i] == chan[P.j]

rows = []
for model in ["hgb_all", "logreg_all", "hgb_noshare"]:
    s_all = S[model].to_numpy()
    m = ~np.isnan(s_all)
    yy, ss, lsm = y[m], s_all[m], ls[m]
    ctx = {"all pairs": (np.zeros(m.sum(), int), False), "same week": (week[m], True),
           "same day": (day[m], True), "same channel": (chan[m], False),
           "same channel x week": (pd.factorize(pd.Series(chan[m]) + "_" + pd.Series(week[m]).astype(str))[0], True),
           "same channel x day": (pd.factorize(pd.Series(chan[m]) + "_" + pd.Series(day[m]).astype(str))[0], True)}
    for name, (g, nested) in ctx.items():
        f = an.within_auc_boot if nested else an.pooled_auc_boot
        est, lo, hi, npairs = f(yy, ss, g, week[m])
        conc, nconc = an.random_pair_concordance(ss, lsm, g, min_gap=np.log(2))
        rows.append(dict(model=model, stratum=name, auc=est, lo=lo, hi=hi, pairs=npairs,
                         conc_2x=conc, conc_pairs=nconc))
    for name, Q in [("same-story twins", P), ("same-story twins, same channel", P[P.same_channel])]:
        Q = Q[m[Q.i] & m[Q.j]]
        a, b = Q.i.to_numpy(), Q.j.to_numpy()
        est, lo, hi, n = an.pair_concordance(s_all, y, a, b, Q.story.to_numpy())
        c2, c2lo, c2hi, n2 = an.pair_concordance(s_all, ls, a, b, Q.story.to_numpy(), min_gap=np.log(2))
        rows.append(dict(model=model, stratum=name, auc=est, lo=lo, hi=hi, pairs=n,
                         conc_2x=c2, conc_lo=c2lo, conc_hi=c2hi, conc_pairs=n2))
    print(model, "done", flush=True)
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s2_anatomy.csv", index=False)
print(R.to_string(index=False))
