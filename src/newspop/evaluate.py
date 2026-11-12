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
