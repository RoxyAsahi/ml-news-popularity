"""S6 (RQ3): semantic memory. Does the recent popularity of similar earlier stories help?

For each article: the 20 earlier-dated articles with the most similar slug (TF-IDF cosine), their
log shares averaged with weights sim * 0.5 ** (age_days / half_life), half-life in {inf, 7, 30, 180}
days (time decay in the spirit of VCPToolBox's TimeDecay; independent implementation). Point-in-time:
only articles published on earlier days are used (their shares are, like kw_*, measured at the
dataset's collection date). Compared with past-only HistGB scores with and without these features
(monthly rolling origin), paired week-block bootstrap.

Output: results/tables/s6_memory.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from newspop.data import ROOT, feature_set, load
from newspop import anatomy as an, stories as st
from newspop.evaluate import paired_bootstrap

df = load()
n = len(df)
y = (df.shares >= 1400).astype(int).to_numpy()
ls = np.log1p(df.shares.to_numpy())
day, week = st.day_index(df), st.week_index(df)
V = st.slug_vectors(df)
K, H = 20, [None, 7, 30, 180]
M = {f"memory_h{h or 'inf'}": np.full(n, np.nan) for h in H}
VT = V.T.tocsc()
for s in range(0, n, 2000):
    S = (V[s:s + 2000] @ VT).toarray()
    for r, i in enumerate(range(s, min(n, s + 2000))):
        prev = np.flatnonzero(day < day[i])
        if len(prev) < K:
            continue
        sim = S[r, prev]
        top = np.argpartition(-sim, K)[:K]
        j, w0 = prev[top], np.maximum(sim[top], 1e-6)
        for h in H:
            w = w0 if h is None else w0 * 0.5 ** ((day[i] - day[j]) / h)
            M[f"memory_h{h or 'inf'}"][i] = (w * ls[j]).sum() / w.sum()
M = pd.DataFrame(M)
base_cols = feature_set(df, "all")
X0 = df[base_cols].to_numpy(float)
X1 = np.c_[X0, M.to_numpy()]
s0 = an.past_only_scores(df, X0, y)
s1 = an.past_only_scores(df, X1, y)
m = ~np.isnan(s0)
pb = paired_bootstrap(y[m], s1[m], s0[m], groups=week[m])
rows = [dict(quantity="AUC with memory minus without (past-only HistGB)", value=pb["diff"], lo=pb["lo"], hi=pb["hi"])]
for c in M:
    ok = m & ~np.isnan(M[c].to_numpy())
    rows.append(dict(quantity=f"Spearman({c}, log shares)", value=spearmanr(M[c][ok], ls[ok])[0]))
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s6_memory.csv", index=False)
print(R.to_string(index=False))
