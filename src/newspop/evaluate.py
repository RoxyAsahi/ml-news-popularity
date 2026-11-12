"""Metrics reported for every classification experiment."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def classification_report(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "pos_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


# ------------------------------------------------------------------ uncertainty
def bootstrap_ci(y_true, score, metric=roc_auc_score, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap (1 - alpha) interval of `metric` on a test set."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y_true[i].min() == y_true[i].max():
            continue
        stats.append(metric(y_true[i], score[i]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, score)), float(lo), float(hi)


def paired_bootstrap(y_true, score_a, score_b, metric=roc_auc_score, n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Difference metric(a) - metric(b) on the SAME resampled test articles.

    Returns the observed difference, its percentile interval and the share of resamples
    in which a beats b. An interval that contains 0 = no detectable difference.
    """
    y_true, a, b = map(np.asarray, (y_true, score_a, score_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y_true[i].min() == y_true[i].max():
            continue
        diffs.append(metric(y_true[i], a[i]) - metric(y_true[i], b[i]))
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(metric(y_true, a) - metric(y_true, b)), "lo": float(lo), "hi": float(hi),
            "p_a_better": float((diffs > 0).mean())}


def paired_fold_diff(values_a, values_b, alpha: float = 0.05) -> dict[str, float]:
    """Mean and t-interval of per-fold differences (e.g. rolling-origin folds)."""
    from scipy import stats
    d = np.asarray(values_a) - np.asarray(values_b)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(1 - alpha / 2, len(d) - 1)
    return {"diff": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "n_folds": len(d)}
