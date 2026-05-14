"""Minimal valid model module.

Demonstrates the agent's contract: fit_model(X, y, problem_type, metric_name)
returns a model with .predict(X) (and .predict_proba for proba-metric paths).
Picks a small sklearn classifier/regressor — the goal is a sanity model, not
a winning one.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


def fit_model(X_train, y_train: np.ndarray, problem_type: str, metric_name: str):
    """Return a fitted classifier or regressor."""
    if problem_type == "classification":
        model = LogisticRegression(max_iter=500, random_state=42)
    elif problem_type == "regression":
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"unsupported problem_type '{problem_type}'")
    model.fit(X_train, y_train)
    return model
