"""Chronological splits. Rows must already be sorted oldest-first (see data.load)."""
from __future__ import annotations

import numpy as np


def temporal_split(n: int, train: float = 0.7, val: float = 0.1):
    """Return index arrays (train, val, test) as contiguous blocks in time."""
    a, b = int(n * train), int(n * (train + val))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def random_split(n: int, test: float = 0.2, seed: int = 0):
    """Random split for the random-vs-temporal comparison (RQ1)."""
    perm = np.random.RandomState(seed).permutation(n)
    cut = int(n * (1 - test))
    return perm[:cut], perm[cut:]


def rolling_origin(n: int, n_folds: int = 5, min_train: float = 0.4):
    """Expanding-window folds: train on [0, t), test on the next block.

    Yields (train_idx, test_idx) pairs; the first fold trains on `min_train` of the data.
    """
    start = int(n * min_train)
    block = (n - start) // n_folds
    for k in range(n_folds):
        t = start + k * block
        end = n if k == n_folds - 1 else t + block
        yield np.arange(t), np.arange(t, end)


def recency_weights(age_days: np.ndarray, half_life: float | None) -> np.ndarray | None:
    """Half-life sample weights 0.5 ** (age / half_life); age measured from the newest
    training article. Idea inspired by the TimeDecay mechanism in VCPToolBox
    (independent implementation)."""
    if half_life is None:
        return None
    age = np.asarray(age_days, dtype=float)
    return 0.5 ** ((age - age.min()) / half_life)
