"""Past-only scores and pairwise (within-group) AUC: where does predictive signal live?

Pairwise AUC within a grouping G = P(score_pos > score_neg) over all (popular, unpopular) pairs that
share the same G value (ties count 1/2). With G = constant it is the ordinary ROC-AUC. Fixing finer
groups (day, channel, channel x day, story) removes the part of the signal that only tells groups
apart, so the drop from one level to the next says how much of the model's skill sits at that level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .models import fit, get_model

HGB_PARAMS = dict(learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, max_iter=500)
LOGREG_PARAMS = dict(penalty="l1", C=0.1)


def past_only_scores(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, model: str = "hgb",
                     first_month: str = "2013-07", seed: int = 0) -> np.ndarray:
    """Monthly rolling origin: for each calendar month m >= first_month, train on every article
    published before m and score the articles of m. Earlier articles get NaN."""
    month = df["publish_date"].dt.to_period("M").astype(str).to_numpy()
    params = HGB_PARAMS if model == "hgb" else LOGREG_PARAMS
    out = np.full(len(df), np.nan)
    for m in sorted(set(month)):
        if m < first_month:
            continue
        tr, te = month < m, month == m
        clf = fit(get_model(model, seed=seed, **params), X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def group_u(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per group: Mann-Whitney U of positives over negatives and the number of pairs."""
    d = pd.DataFrame({"y": y, "s": s, "g": groups})
    d["r"] = d.groupby("g")["s"].rank(method="average")
    agg = d.groupby("g").agg(n=("y", "size"), npos=("y", "sum"))
    agg["rsum"] = d[d.y == 1].groupby("g")["r"].sum().reindex(agg.index).fillna(0.0)
    agg["nneg"] = agg["n"] - agg["npos"]
    agg["u"] = agg["rsum"] - agg["npos"] * (agg["npos"] + 1) / 2
    agg["pairs"] = agg["npos"] * agg["nneg"]
    return agg[agg.pairs > 0][["u", "pairs"]]


def within_auc(y, s, groups) -> tuple[float, float]:
    g = group_u(np.asarray(y), np.asarray(s), np.asarray(groups))
    return float(g.u.sum() / g.pairs.sum()), float(g.pairs.sum())


def within_auc_boot(y, s, groups, clusters, n_boot: int = 500, seed: int = 0):
    """Pairwise AUC within `groups` with a cluster bootstrap over `clusters` (e.g. weeks).
    Groups must be nested in clusters (each group lies in one cluster) -- true for day, week,
    channel x day, channel x week. Returns estimate, 2.5% and 97.5% quantiles, number of pairs."""
    y, s, groups, clusters = map(np.asarray, (y, s, groups, clusters))
    g = group_u(y, s, groups)
    cl = pd.Series(clusters, index=groups).groupby(level=0).first().reindex(g.index)
    per = g.groupby(cl.to_numpy()).sum()
    rng = np.random.default_rng(seed)
    k = len(per)
    U, Pn = per.u.to_numpy(), per.pairs.to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        boots.append(U[pick].sum() / Pn[pick].sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(U.sum() / Pn.sum()), float(lo), float(hi), float(Pn.sum())


def pooled_auc_boot(y, s, groups, clusters, n_boot: int = 300, seed: int = 0):
    """Pairwise AUC within (possibly non-nested) `groups`, bootstrap by resampling whole clusters
    with replacement and recomputing on the stacked rows (slow path, for pooled / channel)."""
    y, s, groups, clusters = map(np.asarray, (y, s, groups, clusters))
    est, npairs = within_auc(y, s, groups)
    labels = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in labels}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[c] for c in rng.choice(labels, len(labels))])
        boots.append(within_auc(y[idx], s[idx], groups[idx])[0])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi), npairs


def pair_concordance(s, target, a, b, clusters, min_gap: float = 0.0, n_boot: int = 1000, seed: int = 0):
    """Share of explicit pairs (a[k], b[k]) whose score order matches the target order, among
    pairs whose target differs by more than `min_gap` (ties in score count 1/2). Cluster bootstrap
    over `clusters` (one label per pair, e.g. the story id)."""
    s, target, a, b, clusters = map(np.asarray, (s, target, a, b, clusters))
    gap = target[a] - target[b]
    keep = np.abs(gap) > min_gap
    ds = s[a] - s[b]
    v = np.where(ds == 0, 0.5, (np.sign(ds) == np.sign(gap)).astype(float))[keep]
    cl = clusters[keep]
    est = float(v.mean())
    labels, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(labels), len(labels))
        boots.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi), int(keep.sum())


def random_pair_concordance(s, target, groups, min_gap: float, n_pairs: int = 2_000_000, seed: int = 0):
    """Concordance on random pairs drawn within the same group (for strata too large to enumerate)."""
    s, target, groups = map(np.asarray, (s, target, groups))
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.r_[0, np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    sizes = ends - starts
    ok = sizes >= 2
    starts, sizes = starts[ok], sizes[ok]
    w = sizes * (sizes - 1) / 2
    gsel = rng.choice(len(starts), n_pairs, p=w / w.sum())
    a = order[starts[gsel] + rng.integers(0, sizes[gsel])]
    b = order[starts[gsel] + rng.integers(0, sizes[gsel])]
    keep = (a != b) & (np.abs(target[a] - target[b]) > min_gap)
    a, b = a[keep], b[keep]
    ds, dt = s[a] - s[b], target[a] - target[b]
    v = np.where(ds == 0, 0.5, (np.sign(ds) == np.sign(dt)).astype(float))
    return float(v.mean()), int(len(v))
