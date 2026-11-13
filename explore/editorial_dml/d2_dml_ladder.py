"""D2: deconfounded effect of each editorial lever, with the confounder set growing step by step.

Ladder of confounder sets W (common.control_sets):
  none   nothing held fixed (raw slope)
  ctx    publication date, day of week, section
  topic  + LDA topics, keyword history (kw_*), number of keywords
  title  + 64 principal components of the real headline's bge vector        <- main specification
  ref    + shares of the older Mashable articles the article links to (a proxy for how hot the story is;
         a descendant of the internal-link lever, so not valid for that one)
  ext    + dictionary sentiment, vocabulary rates                              (may over-control)
  ols    the 'title' set entered linearly with week fixed effects (classical regression adjustment)

For every W: cross-fitted residuals of log shares and of every lever (LightGBM, 5 folds of whole weeks,
several repetitions), then (single) residual-on-residual slope for one lever at a time and (joint) all
levers together in the final regression, so each is read holding the other levers fixed.
Standard errors are clustered by week. The weekend lever uses W without the day of week.

Outputs: out/d2_ladder.csv, out/d2_nuisance_r2.csv
"""
import sys

import numpy as np
import pandas as pd

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
A = [l for l in C.LEVERS if l != "weekend"]
ST = list(C.STYLE)
REPS = {"ctx": 2, "topic": 2, "title": 5, "ref": 2, "ext": 2}
if len(sys.argv) > 1:
    REPS = {b: int(sys.argv[1]) for b in REPS}
dm = lambda a: a - a.mean(0)
est, r2 = [], []


def add(base, rep, mode, names, b, V):
    for i, nme in enumerate(names):
        est.append(dict(spec=base, rep=rep, mode=mode, lever=nme, theta=b[i], se=np.sqrt(V[i, i])))


# no controls
for l in list(C.LEVERS) + ST:
    t = (T[l] if l in T else S[l]).to_numpy()
    b, V = C.cluster_ols(dm(Y), dm(t), wk); add("none", 0, "single", [l], b, V)

# classical linear adjustment: title set + week and weekday dummies, by Frisch-Waugh
from sklearn.linear_model import LinearRegression
for nodow in (False, True):
    Z = [W["title_nodow"][:, 1:], pd.get_dummies(df.week, dtype=float).to_numpy()[:, 1:]]
    if not nodow:
        Z.append(pd.get_dummies(df.dow, dtype=float).to_numpy()[:, 1:])
    Z = np.column_stack(Z)
    Z[:, :] = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    res = lambda v: v - LinearRegression().fit(Z, v).predict(Z)
    ry = res(Y)
    if nodow:
        names = ["weekend"] + A
        R = np.column_stack([res(T[l].to_numpy()) for l in names])
        b, V = C.cluster_ols(ry, R[:, 0], wk); add("ols", 0, "single", ["weekend"], b, V)
        b, V = C.cluster_ols(ry, R, wk); add("ols", 0, "joint", ["weekend"], b[:1], V[:1, :1])
    else:
        R = np.column_stack([res(T[l].to_numpy()) for l in A])
        Rs = np.column_stack([res(S[s].to_numpy()) for s in ST])
        for i, l in enumerate(A):
            b, V = C.cluster_ols(ry, R[:, i], wk); add("ols", 0, "single", [l], b, V)
        for i, s in enumerate(ST):
            b, V = C.cluster_ols(ry, Rs[:, i], wk); add("ols", 0, "single", [s], b, V)
        b, V = C.cluster_ols(ry, R, wk); add("ols", 0, "joint", A, b, V)
        b, V = C.cluster_ols(ry, np.column_stack([R, Rs]), wk)
        add("ols", 0, "joint", ST, b[len(A):], V[len(A):, len(A):])
print("linear done", flush=True)

for base, reps in REPS.items():
    for rep in range(reps):
        # family A: day of week held fixed
        ry = C.residuals(df, "Y", Y, base, rep, W)
        r2.append(dict(spec=base, rep=rep, target="Y", r2=1 - ry.var() / Y.var()))
        R = np.column_stack([C.residuals(df, l, T[l].to_numpy(), base, rep, W) for l in A])
        for i, l in enumerate(A):
            r2.append(dict(spec=base, rep=rep, target=l, r2=1 - R[:, i].var() / T[l].var()))
            b, V = C.cluster_ols(dm(ry), dm(R[:, i]), wk); add(base, rep, "single", [l], b, V)
        b, V = C.cluster_ols(dm(ry), dm(R), wk); add(base, rep, "joint", A, b, V)
        if base != "ext":  # ext already contains the style measures
            Rs = np.column_stack([C.residuals(df, s, S[s].to_numpy(), base, rep, W) for s in ST])
            for i, s in enumerate(ST):
                r2.append(dict(spec=base, rep=rep, target=s, r2=1 - Rs[:, i].var() / S[s].var()))
                b, V = C.cluster_ols(dm(ry), dm(Rs[:, i]), wk); add(base, rep, "single", [s], b, V)
            b, V = C.cluster_ols(dm(ry), dm(np.column_stack([R, Rs])), wk)
            add(base, rep, "joint", ST, b[len(A):], V[len(A):, len(A):])
        # family B: weekend (day of week not in W)
        ryb = C.residuals(df, "Y", Y, base, rep, W, lever="weekend")
        rw = C.residuals(df, "weekend", T.weekend.to_numpy(), base, rep, W, lever="weekend")
        r2.append(dict(spec=base, rep=rep, target="weekend", r2=1 - rw.var() / T.weekend.var()))
        b, V = C.cluster_ols(dm(ryb), dm(rw), wk); add(base, rep, "single", ["weekend"], b, V)
        Rb = np.column_stack([rw] + [T[l].to_numpy() - C.oof(l, T[l].to_numpy(), W[base + "_nodow"], C.week_folds(df, rep),
                                                             f"{base}_nodow__week__r{rep}") for l in A])
        b, V = C.cluster_ols(dm(ryb), dm(Rb), wk); add(base, rep, "joint", ["weekend"], b[:1], V[:1, :1])
        print(base, rep, "R2(Y)", round(1 - ry.var() / Y.var(), 4), flush=True)

E = pd.DataFrame(est)
rows = []
for (spec, mode, lever), gdf in E.groupby(["spec", "mode", "lever"], sort=False):
    th, se = C.combine(gdf.theta.to_numpy(), gdf.se.to_numpy())
    kind = C.LEVERS[lever][1] if lever in C.LEVERS else "sd"
    sd = (T[lever] if lever in T else S[lever]).std()
    rows.append(dict(spec=spec, mode=mode, lever=lever, label=(C.LEVERS.get(lever) or C.STYLE[lever])[0], theta=th, se=se,
                     lo=th - 1.96 * se, hi=th + 1.96 * se, z=th / se, reps=len(gdf),
                     per_sd=th * sd, per_sd_lo=(th - 1.96 * se) * sd, per_sd_hi=(th + 1.96 * se) * sd,
                     pct=C.pct(th, kind) if kind != "sd" else 100 * (np.exp(th * sd) - 1),
                     pct_lo=C.pct(th - 1.96 * se, kind) if kind != "sd" else 100 * (np.exp((th - 1.96 * se) * sd) - 1),
                     pct_hi=C.pct(th + 1.96 * se, kind) if kind != "sd" else 100 * (np.exp((th + 1.96 * se) * sd) - 1)))
L = pd.DataFrame(rows)
L.round(5).to_csv(C.OUT / "d2_ladder.csv", index=False)
N = pd.DataFrame(r2).groupby(["spec", "target"], sort=False).r2.mean().unstack(0)
N.round(4).to_csv(C.OUT / "d2_nuisance_r2.csv")
print(N.round(3).to_string())
P = L[L["mode"] == "single"].pivot(index="lever", columns="spec", values="pct").reindex(list(C.LEVERS) + ST)
print("\n% change in shares (doubling / weekend / +1 SD for style), one lever at a time\n", P[["none", "ctx", "topic", "title", "ref", "ext", "ols"]].round(1).to_string())
Pj = L[L["mode"] == "joint"].pivot(index="lever", columns="spec", values="pct").reindex(list(C.LEVERS) + ST)
print("\nsame, all levers jointly\n", Pj[["ctx", "topic", "title", "ref", "ext", "ols"]].round(1).to_string())
M = L[(L.spec == "title")].pivot(index="lever", columns="mode", values=["pct", "pct_lo", "pct_hi", "z"]).reindex(list(C.LEVERS) + ST)
print("\nmain spec\n", M.round(2).to_string())

# how much of what W leaves unexplained do the levers account for, all together? (main spec, repetition 0)
ry = dm(C.residuals(df, "Y", Y, "title", 0, W))
RA = dm(np.column_stack([C.residuals(df, l, T[l].to_numpy(), "title", 0, W) for l in A]))
art = [i for i, l in enumerate(A) if l not in C.DAY_LEVEL]; dayl = [i for i, l in enumerate(A) if l in C.DAY_LEVEL]
r2 = lambda X: 1 - ((ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]) ** 2).sum() / (ry ** 2).sum()
B = pd.DataFrame([dict(block="six article levers", partial_r2=r2(RA[:, art])), dict(block="two crowding counts", partial_r2=r2(RA[:, dayl])),
                  dict(block="all eight", partial_r2=r2(RA)), dict(block="W itself (out-of-fold R2 of log shares)", partial_r2=1 - ry.var() / Y.var())])
B.round(4).to_csv(C.OUT / "d2_joint_r2.csv", index=False); print(B.round(4).to_string(index=False))
