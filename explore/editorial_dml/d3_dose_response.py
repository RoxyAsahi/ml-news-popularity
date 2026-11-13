"""D3: is there a threshold? Each lever cut into bins, one bin as the reference, main confounder set.

Partially linear model with one dummy per bin: every dummy is residualised on W (cross-fitted, whole-week
folds) and log shares are regressed on the dummies' residuals, (single) alone or (joint) together with
the residuals of the other levers, so 'two videos instead of none' is read at fixed length, images, links.

Output: out/d3_dose_response.csv
"""
import numpy as np
import pandas as pd

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
A = [l for l in C.LEVERS if l != "weekend"]
BINS = {  # lever -> (raw column, [(label, lo, hi)], reference label)
    "imgs": (df.num_imgs, [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-9", 4, 9), ("10+", 10, 1e9)], "1"),
    "videos": (df.num_videos, [("0", 0, 0), ("1", 1, 1), ("2+", 2, 1e9)], "0"),
    "length": (df.n_tokens_content, [("<250", 0, 249), ("250-449", 250, 449), ("450-749", 450, 749),
                                     ("750-1199", 750, 1199), ("1200+", 1200, 1e9)], "250-449"),
    "links_self": (df.num_self_hrefs, [("0", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5+", 5, 1e9)], "1-2"),
    "links_out": ((df.num_hrefs - df.num_self_hrefs).clip(lower=0), [("0-2", 0, 2), ("3-6", 3, 6), ("7-12", 7, 12), ("13+", 13, 1e9)], "3-6"),
    "title_len": (df.n_tokens_title, [("<=8", 0, 8), ("9-11", 9, 11), ("12+", 12, 1e9)], "9-11"),
}
REPS = 2
dm = lambda a: a - a.mean(0)
est = []
for rep in range(REPS):
    ry = dm(C.residuals(df, "Y", Y, "title", rep, W))
    R = {l: C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in A}
    for lv, (col, bins, ref) in BINS.items():
        names = [b[0] for b in bins if b[0] != ref]
        D = np.column_stack([C.residuals(df, f"{lv}_bin{b[0]}", ((col >= b[1]) & (col <= b[2])).to_numpy(float), "title", rep, W)
                             for b in bins if b[0] != ref])
        others = np.column_stack([R[l] for l in A if l != lv])
        for mode, Xm in [("single", D), ("joint", np.column_stack([D, others]))]:
            b, V = C.cluster_ols(ry, dm(Xm), wk)
            for i, nme in enumerate(names):
                est.append(dict(lever=lv, bin=nme, mode=mode, rep=rep, theta=b[i], se=np.sqrt(V[i, i])))
        print(rep, lv, flush=True)
E = pd.DataFrame(est)
rows = []
for lv, (col, bins, ref) in BINS.items():
    for lab, lo, hi in bins:
        n = int(((col >= lo) & (col <= hi)).sum())
        for mode in ("single", "joint"):
            if lab == ref:
                rows.append(dict(lever=lv, bin=lab, mode=mode, n=n, theta=0.0, se=0.0, pct=0.0, pct_lo=0.0, pct_hi=0.0, reference=True)); continue
            gdf = E[(E.lever == lv) & (E.bin == lab) & (E["mode"] == mode)]
            th, se = C.combine(gdf.theta.to_numpy(), gdf.se.to_numpy())
            rows.append(dict(lever=lv, bin=lab, mode=mode, n=n, theta=th, se=se, pct=100 * (np.exp(th) - 1),
                             pct_lo=100 * (np.exp(th - 1.96 * se) - 1), pct_hi=100 * (np.exp(th + 1.96 * se) - 1), reference=False))
R = pd.DataFrame(rows)
R.round(4).to_csv(C.OUT / "d3_dose_response.csv", index=False)
print(R[R["mode"] == "joint"].round(2).to_string(index=False))
