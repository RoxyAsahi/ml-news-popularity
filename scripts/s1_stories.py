"""S1: same-story twin pairs from URL slugs: counts by threshold, story sizes, annotation sample.

Outputs
  results/tables/s1_twin_counts.csv     pairs / articles / same-channel share by similarity threshold
  results/tables/s1_story_sizes.csv     size distribution of stories (connected components) at 0.5
  data/annotations/twins_sample.csv     30 pairs per similarity band for manual annotation
                                        (written only if absent, so annotations are never overwritten)
"""
import numpy as np
import pandas as pd

from newspop.data import ROOT, load
from newspop import stories as st

TAB = ROOT / "results" / "tables"
ANN = ROOT / "data" / "annotations" / "twins_sample.csv"

df = load()
P = st.twin_pairs(df, min_sim=0.3)
slug = st.slug_text(df["url"])
chan = st.channel(df)

rows = []
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    Q = P[P.sim >= th]
    rows.append({"min_sim": th, "pairs": len(Q), "articles": len(set(Q.i) | set(Q.j)),
                 "same_channel_share": round(Q.same_channel.mean(), 3),
                 "median_days_apart": float(Q.dt.median())})
counts = pd.DataFrame(rows)
counts.to_csv(TAB / "s1_twin_counts.csv", index=False)
print(counts.to_string(index=False))

sid = st.story_ids(len(df), P[P.sim >= st.SIM_THRESHOLD])
sizes = pd.Series(sid).value_counts()
sz = sizes.value_counts().sort_index().rename_axis("story_size").reset_index(name="n_stories")
sz.to_csv(TAB / "s1_story_sizes.csv", index=False)
print("largest stories:", sizes.head(5).tolist())
big = sizes.index[0]
print("example of the largest story:", slug[sid == big].head(8).tolist())

if not ANN.exists():
    bands = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    rng = np.random.default_rng(2026)
    parts = []
    for lo, hi in bands:
        Q = P[(P.sim >= lo) & (P.sim < hi)]
        parts.append(Q.iloc[rng.choice(len(Q), 30, replace=False)].assign(band=f"[{lo},{min(hi,1.0)})"))
    S = pd.concat(parts).sample(frac=1, random_state=1)  # shuffled so bands are not visible in order
    out = pd.DataFrame({
        "pair_id": range(len(S)), "i": S.i.values, "j": S.j.values,
        "date_i": df.publish_date.dt.date.values[S.i], "date_j": df.publish_date.dt.date.values[S.j],
        "channel_i": chan[S.i], "channel_j": chan[S.j], "slug_i": slug.values[S.i], "slug_j": slug.values[S.j],
        "sim": S.sim.round(3).values, "band": S.band.values, "label": ""})
    out.to_csv(ANN, index=False)
    print("wrote", ANN, len(out))
