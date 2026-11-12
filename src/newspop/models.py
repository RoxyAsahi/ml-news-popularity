"""Model zoo. Every model is a Pipeline so preprocessing is fit on training data only."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def signed_log1p(X):
    """log1p that keeps the sign; tames the heavy-tailed count features."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def _scaled(model) -> Pipeline:
    return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(), model)


def get_model(name: str, seed: int = 0, **params):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        l1 = params.get("penalty", "l2") == "l1"
        return _scaled(LogisticRegression(max_iter=5000, C=params.get("C", 1.0),
                                          penalty="l1" if l1 else "l2",
                                          solver="liblinear" if l1 else "lbfgs",
                                          random_state=seed))
    if name == "knn":
        return _scaled(KNeighborsClassifier(n_neighbors=params.get("k", 50)))
    if name == "rf":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 300),
                                      min_samples_leaf=5, n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"unknown model {name!r}")


def fit(model, X, y, sample_weight=None):
    """Fit, routing sample_weight to the final estimator of a Pipeline."""
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)
