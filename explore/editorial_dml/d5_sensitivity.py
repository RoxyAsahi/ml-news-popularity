"""D5: how fragile are the lever effects? Everything here targets the one assumption DML cannot test:
that nothing outside W drives both the lever and the shares (no unobserved confounding).

(a) Cinelli & Hazlett (2020) on the final residual-on-residual regression of the main specification
    (the partially linear case of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022): robustness
    value RV (share of the remaining variance of BOTH lever and shares a hidden factor must explain to
    erase the effect), RV for losing 5% significance, and benchmarks: the estimate that would remain if a
    hidden factor were as strong as an entire observed block (keyword history, section + LDA topics,
    headline vector, date), and how many times stronger than the block it must be to erase the effect.
    Block strength = drop of out-of-fold residual variance when the block is removed from W.
(b) Negative-control outcome: the shares of the older Mashable articles an article links to cannot be
    caused by its images, length, ...; if a lever 'moves' them, W misses how hot the story is.
(c) Placebo lever: the same section's volume seven days later.
(d) Cross-fitting on random folds instead of whole weeks; a different learner (extra trees).
(e) Triangulation with same-story twin differences (pairs rebuilt here on the realigned urls).

Outputs: out/d5_sensitivity.csv, out/d5_negative_control.csv, out/d5_robustness.csv, out/d5_twins.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
A = [l for l in C.LEVERS if l != "weekend"]
ART = ["imgs", "videos", "length", "title_len", "links_out", "links_self"]
dm = lambda a: a - a.mean(0)
BLOCKS = {"kw": "keyword history", "ldachan": "section + LDA topics", "emb": "headline vector", "time": "publication date"}

# ---------------------------------------------------------------- (a) Cinelli-Hazlett
rows = []
for lv, (label, kind) in C.LEVERS.items():
    ry = dm(C.residuals(df, "Y", Y, "title", 0, W, lever=lv)); rt = dm(C.residuals(df, lv, T[lv].to_numpy(), "title", 0, W, lever=lv))
    b, V = C.cluster_ols(ry, rt, wk)
    th, se = b[0], np.sqrt(V[0, 0])
    G = len(np.unique(wk)); dof = G - 1                      # conservative: clusters, not articles
    f = abs(th / se) / np.sqrt(dof)                          # partial Cohen's f with cluster-robust t
    f_iid = abs(th) * rt.std() / (ry - th * rt).std()        # plain partial f (what RV is defined on)
    r2yd = f_iid ** 2 / (1 + f_iid ** 2)
    rv = 0.5 * (np.sqrt(f_iid ** 4 + 4 * f_iid ** 2) - f_iid ** 2)
    fa = max(0.0, f_iid - 1.96 * se / abs(th) * f_iid)       # shrink by the cluster-robust margin of error
    rva = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2)
    row = dict(lever=lv, label=label, theta=th, se=se, pct=C.pct(th, kind), partial_r2_lever_shares=r2yd, RV=rv, RV_5pct=rva)
    scale = (ry - th * rt).std() / rt.std()
    for blk, blab in BLOCKS.items():
        ry0 = C.residuals(df, "Y", Y, f"title-{blk}", 0, W, lever=lv); rt0 = C.residuals(df, lv, T[lv].to_numpy(), f"title-{blk}", 0, W, lever=lv)
        r2y = max(0.0, 1 - ry.var() / ry0.var()); r2d = max(0.0, 1 - rt.var() / rt0.var())
        fy, fd = r2y / (1 - r2y), r2d / (1 - r2d)             # strength of the block, as f^2

        def bias(k):
            r2dz, r2yz = min(k * fd, 0.999), min(k * fy, 1.0)
            return np.sqrt(r2yz * r2dz / (1 - r2dz)) * scale
        adj = np.sign(th) * (abs(th) - bias(1.0))
        ks = np.linspace(0.01, 200, 20000); hit = ks[np.array([bias(k) for k in ks]) >= abs(th)]
        row.update({f"{blk}_r2_shares": r2y, f"{blk}_r2_lever": r2d, f"{blk}_adj_pct": C.pct(adj, kind),
                    f"{blk}_times_to_erase": hit[0] if len(hit) else np.inf})
    rows.append(row); print(lv, "RV", round(rv, 4), flush=True)
SA = pd.DataFrame(rows); SA.round(4).to_csv(C.OUT / "d5_sensitivity.csv", index=False)
print(SA[["lever", "pct", "partial_r2_lever_shares", "RV", "RV_5pct"] + [c for c in SA if c.endswith("adj_pct") or c.endswith("erase")]].round(3).to_string(index=False))

# ---------------------------------------------------------------- (b) negative-control outcome, (c) placebo lever
nc = np.log1p(df.self_reference_avg_sharess.to_numpy()); m = (df.num_self_hrefs > 0).to_numpy() & (nc > 0)
rows = []
for rep in range(2):
    fo = C.week_folds(df, rep)
    rn = np.full(len(Y), np.nan)
    f = C.CACHE / f"title__week__r{rep}__NCsub.npy"
    if f.exists():
        rn[m] = np.load(f)
    else:
        pred = np.zeros(m.sum()); Xs, ys, fs = W["title"][m], nc[m], fo[m]
        for k in range(C.N_FOLDS):
            pred[fs == k] = C.make_learner(seed=k).fit(Xs[fs != k], ys[fs != k]).predict(Xs[fs == k])
        np.save(f, ys - pred); rn[m] = ys - pred
    ry = C.residuals(df, "Y", Y, "title", rep, W)
    R = np.column_stack([C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in ART])
    for i, l in enumerate(ART):
        for out, yy in [("linked articles' shares (negative control)", rn), ("own shares, same articles", ry)]:
            b, V = C.cluster_ols(dm(yy[m]), dm(R[m, i]), wk[m])
            rows.append(dict(lever=l, outcome=out, mode="single", rep=rep, theta=b[0], se=np.sqrt(V[0, 0])))
    for out, yy in [("linked articles' shares (negative control)", rn), ("own shares, same articles", ry)]:
        b, V = C.cluster_ols(dm(yy[m]), dm(R[m]), wk[m])
        for i, l in enumerate(ART):
            rows.append(dict(lever=l, outcome=out, mode="joint", rep=rep, theta=b[i], se=np.sqrt(V[i, i])))
    # placebo: next week's volume
    lead = df.vol_same_lead7.to_numpy(); ok = np.isfinite(lead)
    rl = np.full(len(Y), np.nan)
    f = C.CACHE / f"title__week__r{rep}__lead7sub.npy"
    if f.exists():
        rl[ok] = np.load(f)
    else:
        pred = np.zeros(ok.sum()); Xs, ys, fs = W["title"][ok], lead[ok], fo[ok]
        for k in range(C.N_FOLDS):
            pred[fs == k] = C.make_learner(seed=k).fit(Xs[fs != k], ys[fs != k]).predict(Xs[fs == k])
        np.save(f, ys - pred); rl[ok] = ys - pred
    rv_ = C.residuals(df, "vol_same", T.vol_same.to_numpy(), "title", rep, W)
    b, V = C.cluster_ols(dm(ry[ok]), dm(rl[ok]), wk[ok])
    rows.append(dict(lever="vol_same in 7 days (placebo)", outcome="own shares", mode="single", rep=rep, theta=b[0], se=np.sqrt(V[0, 0])))
    b, V = C.cluster_ols(dm(ry[ok]), dm(np.column_stack([rv_[ok], rl[ok]])), wk[ok])
    rows.append(dict(lever="vol_same in 7 days (placebo)", outcome="own shares", mode="joint", rep=rep, theta=b[1], se=np.sqrt(V[1, 1])))
    rows.append(dict(lever="vol_same", outcome="own shares", mode="joint", rep=rep, theta=b[0], se=np.sqrt(V[0, 0])))
E = pd.DataFrame(rows); out = []
for key, gdf in E.groupby(["lever", "outcome", "mode"], sort=False):
    th, se = C.combine(gdf.theta.to_numpy(), gdf.se.to_numpy())
    out.append(dict(lever=key[0], outcome=key[1], mode=key[2], theta=th, se=se, z=th / se, pct=C.pct(th, "log"),
                    pct_lo=C.pct(th - 1.96 * se, "log"), pct_hi=C.pct(th + 1.96 * se, "log"), n=int(m.sum()) if "linked" in key[1] or "same articles" in key[1] else int(ok.sum())))
NCt = pd.DataFrame(out); NCt.round(4).to_csv(C.OUT / "d5_negative_control.csv", index=False)
print(NCt.round(3).to_string(index=False))

# ---------------------------------------------------------------- (d) other folds, other learner
rows = []
from sklearn.ensemble import ExtraTreesRegressor


def et_oof(name, target, X, folds, tag):
    f = C.CACHE / f"{tag}__ET__{name}.npy"
    if f.exists():
        return np.load(f)
    pred = np.zeros(len(target))
    for k in np.unique(folds):
        pred[folds == k] = ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.33, n_jobs=C.THREADS, random_state=int(k)) \
            .fit(X[folds != k], target[folds != k]).predict(X[folds == k])
    np.save(f, pred); return pred


for variant in ("week folds, LightGBM (main)", "random folds, LightGBM", "week folds, extra trees"):
    for lv in C.LEVERS:
        if variant.startswith("random") and lv in C.DAY_LEVEL:
            continue  # day-level levers are perfectly predictable from the date under random folds
        base = C.wname("title", lv); t = T[lv].to_numpy()
        if "extra" in variant:
            fo = C.week_folds(df, 0)
            ry = Y - et_oof("Y", Y, W[base], fo, f"{base}__week__r0"); rt = t - et_oof(lv, t, W[base], fo, f"{base}__week__r0")
        else:
            kind = "random" if variant.startswith("random") else "week"
            ry = C.residuals(df, "Y", Y, "title", 0, W, lever=lv, folds=kind); rt = C.residuals(df, lv, t, "title", 0, W, lever=lv, folds=kind)
        b, V = C.cluster_ols(dm(ry), dm(rt), wk); k = C.LEVERS[lv][1]
        rows.append(dict(variant=variant, lever=lv, theta=b[0], se=np.sqrt(V[0, 0]), pct=C.pct(b[0], k), r2_shares=1 - ry.var() / Y.var(), r2_lever=1 - rt.var() / t.var()))
    print(variant, flush=True)
RB = pd.DataFrame(rows); RB.round(4).to_csv(C.OUT / "d5_robustness.csv", index=False)
print(RB.pivot(index="lever", columns="variant", values="pct").reindex(list(C.LEVERS)).round(1).to_string())
print(RB.pivot(index="lever", columns="variant", values="r2_lever").reindex(list(C.LEVERS)).round(3).to_string())

# ---------------------------------------------------------------- (e) same-story twins
# A second identification strategy that needs no W at all: pairs of articles on the same news story
# (slug cosine >= 0.5, <= 7 days apart, recurring series removed; newspop.stories on the realigned urls).
# Within a pair the story is fixed, so the difference in log shares is regressed on the differences in
# the levers (earlier minus later, with a constant for the earlier-article advantage); story-clustered.
from newspop import stories as st

P = st.clean_twins(df)
later = np.where(P["first"] == P.i, P.j, P.i); first = P["first"].to_numpy()
story = st.story_ids(len(df), P)[first]
tw_names = ART + ["vol_same", "vol_other", "weekend"]
D = T[tw_names].to_numpy()[first] - T[tw_names].to_numpy()[later]
dy = Y[first] - Y[later]
bt, Vt = C.cluster_ols(dy, np.column_stack([D, np.ones(len(dy))]), story)
L = pd.read_csv(C.OUT / "d2_ladder.csv")
J = L[(L.spec == "title") & (L["mode"] == "joint")].set_index("lever"); N0 = L[L.spec == "none"].set_index("lever")
rows = []
for i, l in enumerate(tw_names):
    k = C.LEVERS[l][1]; se = np.sqrt(Vt[i, i])
    rows.append(dict(lever=l, raw_pct=N0.pct[l], dml_pct=J.pct[l], dml_lo=J.pct_lo[l], dml_hi=J.pct_hi[l],
                     twin_pct=C.pct(bt[i], k), twin_lo=C.pct(bt[i] - 1.96 * se, k), twin_hi=C.pct(bt[i] + 1.96 * se, k),
                     twin_pairs=len(P), twin_sd_of_difference=D[:, i].std()))
TW = pd.DataFrame(rows); TW.round(3).to_csv(C.OUT / "d5_twins.csv", index=False)
print(len(P), "twin pairs,", len(np.unique(story)), "stories"); print(TW.round(2).to_string(index=False))
