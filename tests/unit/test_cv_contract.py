"""Unit tests for kaggle_slayer.harness.cv.train_cv.

These tests use minimal hand-written fe/model modules created via tmp_path
to validate the contract behaviour. The richer end-to-end test using the
proper stub modules lives in tests/integration/test_cv_with_stubs.py.
"""

from __future__ import annotations

import textwrap

import pytest

from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness.registry import cv_strategies, metrics


@pytest.fixture
def fe_pass_through(tmp_path):
    """Trivial FE: drops categoricals, passes numerics through."""
    p = tmp_path / "fe.py"
    p.write_text(textwrap.dedent("""
        import pandas as pd

        class _PassThrough:
            def __init__(self, numeric_cols):
                self.numeric_cols = numeric_cols
            def transform(self, df):
                return df[self.numeric_cols].copy()

        def fit_feature_transformer(train_df, target_col):
            numeric = [c for c in train_df.columns
                       if c != target_col and train_df[c].dtype.kind in "fiub"]
            return _PassThrough(numeric)
    """))
    return p


@pytest.fixture
def model_logreg(tmp_path):
    """Trivial model: logistic regression for classification."""
    p = tmp_path / "model.py"
    p.write_text(textwrap.dedent("""
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge

        def fit_model(X_train, y_train, problem_type, metric_name):
            if problem_type == "classification":
                m = LogisticRegression(max_iter=500)
            else:
                m = Ridge()
            m.fit(X_train, y_train)
            return m
    """))
    return p


def test_train_cv_runs_and_returns_cvresult(
    fe_pass_through, model_logreg, synthetic_binary
):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3, random_state=42)
    metric = metrics.get("accuracy")

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert isinstance(result, cv_mod.CVResult)
    assert len(result.fold_scores) == 3
    assert 0.0 <= result.mean <= 1.0
    # Synthetic binary should be well above chance
    assert result.mean > 0.7
    assert result.oof.shape[0] == len(train)


def test_train_cv_with_proba_metric(fe_pass_through, model_logreg, synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("auc")  # needs_proba=True

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert 0.5 < result.mean <= 1.0
    # OOF should be probabilities, not class labels
    assert ((result.oof >= 0.0) & (result.oof <= 1.0)).all()


def test_train_cv_regression(fe_pass_through, model_logreg, synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.get("kfold", n_splits=3)
    metric = metrics.get("rmse")

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert result.mean > 0.0  # rmse is non-negative
    assert result.oof.shape[0] == len(train)


def test_train_cv_rejects_row_dropping_fe(tmp_path, model_logreg, synthetic_binary):
    """Anti-cheat: FE that drops rows must be rejected."""
    fe = tmp_path / "fe.py"
    fe.write_text(textwrap.dedent("""
        import pandas as pd

        class _Dropper:
            def __init__(self, numeric_cols):
                self.numeric_cols = numeric_cols
            def transform(self, df):
                # Drops half the rows — would change the val split
                return df[self.numeric_cols].iloc[::2].copy()

        def fit_feature_transformer(train_df, target_col):
            numeric = [c for c in train_df.columns
                       if c != target_col and train_df[c].dtype.kind in "fiub"]
            return _Dropper(numeric)
    """))
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="rows changed"):
        cv_mod.train_cv(
            fe_path=fe,
            model_path=model_logreg,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_rejects_fe_missing_fit_function(tmp_path, model_logreg, synthetic_binary):
    """If fe.py is missing fit_feature_transformer, raise clearly."""
    fe = tmp_path / "fe.py"
    fe.write_text("# empty\n")
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="fit_feature_transformer"):
        cv_mod.train_cv(
            fe_path=fe,
            model_path=model_logreg,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_rejects_model_missing_fit_function(tmp_path, fe_pass_through, synthetic_binary):
    """If model.py is missing fit_model, raise clearly."""
    model = tmp_path / "model.py"
    model.write_text("# empty\n")
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="fit_model"):
        cv_mod.train_cv(
            fe_path=fe_pass_through,
            model_path=model,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_problem_type_inference_from_metric(
    fe_pass_through, model_logreg, synthetic_binary
):
    """Probability-needing metrics imply classification; train_cv passes the
    right `problem_type` to the agent's fit_model."""
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("auc")
    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )
    # If problem_type was misinferred as regression, Ridge would not have
    # predict_proba and the call would have raised.
    assert result.mean > 0.5


def test_train_cv_multi_class_proba(tmp_path):
    """Multi-class classification with a needs_proba metric (logloss)
    must work end-to-end."""
    import textwrap
    import pandas as pd
    import numpy as np

    fe = tmp_path / "fe.py"
    fe.write_text(textwrap.dedent('''
        import pandas as pd

        class _PT:
            def __init__(self, cols):
                self.cols = cols
            def transform(self, df):
                return df[self.cols].copy()

        def fit_feature_transformer(train_df, target_col):
            cols = [c for c in train_df.columns
                    if c != target_col and train_df[c].dtype.kind in "fiub"]
            return _PT(cols)
    '''))
    model = tmp_path / "model.py"
    model.write_text(textwrap.dedent('''
        from sklearn.linear_model import LogisticRegression

        def fit_model(X_train, y_train, problem_type, metric_name):
            m = LogisticRegression(max_iter=500)
            m.fit(X_train, y_train)
            return m
    '''))
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "target": rng.integers(0, 3, size=n),  # 3-class
    })

    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("logloss")

    result = cv_mod.train_cv(
        fe_path=fe,
        model_path=model,
        train_df=df,
        target_col="target",
        cv=cv,
        metric=metric,
    )
    # Three classes → OOF should be (n, 3)
    assert result.oof.shape == (n, 3)
    # Probabilities should each be in [0, 1] and rows sum ~= 1
    assert ((result.oof >= 0.0) & (result.oof <= 1.0)).all()
    np.testing.assert_allclose(result.oof.sum(axis=1), 1.0, atol=1e-6)
    # logloss is finite and non-negative
    assert result.mean > 0.0
    assert np.isfinite(result.mean)
