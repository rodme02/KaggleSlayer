"""Tests for kaggle_slayer.agent.handlers.ml."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""


def _write_stub_fe(workspace: Workspace) -> None:
    """Mean-impute numerics, drop categoricals."""
    workspace.fe_path.write_text(
        "import pandas as pd\n"
        "\n"
        "class _PT:\n"
        "    def __init__(self, cols, means):\n"
        "        self.cols = cols\n"
        "        self.means = means\n"
        "    def transform(self, df):\n"
        "        out = pd.DataFrame(index=df.index)\n"
        "        for c in self.cols:\n"
        "            if c in df.columns:\n"
        "                out[c] = df[c].fillna(self.means.get(c, 0.0))\n"
        "        return out\n"
        "\n"
        "def fit_feature_transformer(train_df, target_col):\n"
        "    cols = [c for c in train_df.columns if c != target_col and train_df[c].dtype.kind in 'fiub']\n"
        "    means = {c: float(train_df[c].mean()) for c in cols}\n"
        "    return _PT(cols, means)\n"
    )


def _write_stub_model(workspace: Workspace) -> None:
    workspace.model_path.write_text(
        "from sklearn.linear_model import LogisticRegression, Ridge\n"
        "\n"
        "def fit_model(X_train, y_train, problem_type, metric_name):\n"
        "    if problem_type == 'classification':\n"
        "        m = LogisticRegression(max_iter=500, random_state=42)\n"
        "    else:\n"
        "        m = Ridge(alpha=1.0, random_state=42)\n"
        "    m.fit(X_train, y_train)\n"
        "    return m\n"
    )


@pytest.fixture
def comp_ctx(tmp_path):
    """A workspace with a small synthetic binary-classification dataset wired in."""
    import numpy as np

    ws = Workspace.create(root=tmp_path / "comp")
    rng = np.random.default_rng(0)
    n_train, n_test = 200, 50
    train_df = pd.DataFrame({
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train),
        "target": rng.integers(0, 2, size=n_train),
    })
    train_df.to_csv(ws.raw_dir / "train.csv", index=False)
    test_df = pd.DataFrame({
        "id": range(n_test),
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test),
    })
    test_df.to_csv(ws.raw_dir / "test.csv", index=False)
    _write_stub_fe(ws)
    _write_stub_model(ws)
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_set_cv_records_override(comp_ctx):
    result = ml_h.set_cv(comp_ctx, kind="stratified_kfold", n_splits=3)
    assert "stratified_kfold" in result
    assert comp_ctx.cv_kind == "stratified_kfold"
    assert comp_ctx.cv_params == {"n_splits": 3}


def test_set_cv_with_group_col(comp_ctx):
    result = ml_h.set_cv(comp_ctx, kind="group_kfold", n_splits=3, group_col="x1")
    assert "group_kfold" in result
    assert comp_ctx.cv_params == {"n_splits": 3, "group_col": "x1"}


def test_set_cv_rejects_unknown_kind(comp_ctx):
    with pytest.raises(ToolError, match="unknown CV kind"):
        ml_h.set_cv(comp_ctx, kind="random_split")


def test_set_cv_group_kfold_requires_group_col(comp_ctx):
    """F8: kind='group_kfold' is meaningless without a group column —
    reject early with a hint instead of letting it fail at split time."""
    with pytest.raises(ToolError, match="group_col"):
        ml_h.set_cv(comp_ctx, kind="group_kfold", n_splits=3)


def test_train_cv_runs_and_returns_summary(comp_ctx):
    result = ml_h.train_cv(comp_ctx)
    # Result is a string the LLM can read; should mention fold scores and mean
    assert "mean=" in result.lower() or "mean " in result.lower()
    assert "0." in result  # some score value


def test_train_cv_archives_fe_and_model_to_versions(comp_ctx):
    ml_h.train_cv(comp_ctx)
    assert (comp_ctx.workspace.versions_dir / "fe_v01.py").exists()
    assert (comp_ctx.workspace.versions_dir / "model_v01.py").exists()
    # Re-running increments
    ml_h.train_cv(comp_ctx)
    assert (comp_ctx.workspace.versions_dir / "fe_v02.py").exists()
    assert (comp_ctx.workspace.versions_dir / "model_v02.py").exists()


def test_train_cv_uses_cv_override_when_set(comp_ctx):
    """If set_cv was called, train_cv must use that strategy.

    Default for classification is stratified_kfold; explicit override to plain
    kfold should be honored. We assert on the precise strategy name and the
    n_splits the override specified — substring matches won't work because
    "kfold" is a substring of "stratified_kfold".
    """
    ml_h.set_cv(comp_ctx, kind="kfold", n_splits=3)
    result = ml_h.train_cv(comp_ctx)
    assert "kfold (3 folds)" in result
    assert "stratified" not in result


def test_train_cv_lints_fe_and_model_first(comp_ctx):
    """A fe.py calling subprocess must be rejected before load.
    No version archive should be written for a lint-failed module."""
    comp_ctx.workspace.fe_path.write_text(
        "import subprocess\n"
        "\n"
        "class _T:\n"
        "    def transform(self, df):\n"
        "        return df\n"
        "\n"
        "def fit_feature_transformer(train_df, target_col):\n"
        "    subprocess.run(['echo', 'bad'])\n"
        "    return _T()\n"
    )
    with pytest.raises(ToolError, match=r"lint|subprocess"):
        ml_h.train_cv(comp_ctx)
    # Lint failure must not bump the version counter (F7)
    assert not (comp_ctx.workspace.versions_dir / "fe_v01.py").exists()
    assert not (comp_ctx.workspace.versions_dir / "model_v01.py").exists()


def test_submit_local_lints_fe_and_model_first(comp_ctx):
    """A model.py with a disallowed call must be rejected before submit_local
    tries to load it."""
    comp_ctx.workspace.model_path.write_text(
        "import os\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "\n"
        "def fit_model(X_train, y_train, problem_type, metric_name):\n"
        "    os.system('echo bad')\n"
        "    m = LogisticRegression(max_iter=500, random_state=42)\n"
        "    m.fit(X_train, y_train)\n"
        "    return m\n"
    )
    with pytest.raises(ToolError, match=r"lint|os\.system"):
        ml_h.submit_local(comp_ctx, label="x")


def test_train_cv_does_not_archive_when_cv_fails(comp_ctx):
    """If cv_mod.train_cv raises (e.g. fe.transform mutates row count),
    the version archive must not be written. The next attempt should
    still get fe_v01.py."""
    comp_ctx.workspace.fe_path.write_text(
        "import pandas as pd\n"
        "\n"
        "class _Dropper:\n"
        "    def __init__(self, cols):\n"
        "        self.cols = cols\n"
        "    def transform(self, df):\n"
        "        return df[self.cols].iloc[::2].copy()\n"
        "\n"
        "def fit_feature_transformer(train_df, target_col):\n"
        "    cols = [c for c in train_df.columns if c != target_col and train_df[c].dtype.kind in 'fiub']\n"
        "    return _Dropper(cols)\n"
    )
    with pytest.raises(Exception):  # noqa: B017 — CVError or similar
        ml_h.train_cv(comp_ctx)
    assert not (comp_ctx.workspace.versions_dir / "fe_v01.py").exists()
    assert not (comp_ctx.workspace.versions_dir / "model_v01.py").exists()


def test_train_cv_missing_fe_raises(comp_ctx):
    comp_ctx.workspace.fe_path.unlink()
    with pytest.raises(ToolError, match="fe.py"):
        ml_h.train_cv(comp_ctx)


def test_train_cv_missing_model_raises(comp_ctx):
    comp_ctx.workspace.model_path.unlink()
    with pytest.raises(ToolError, match="model.py"):
        ml_h.train_cv(comp_ctx)


def test_submit_local_writes_submission_csv(comp_ctx):
    result = ml_h.submit_local(comp_ctx, label="lr_baseline")
    # Check file exists in submissions/ and the result message references it
    submissions = list(comp_ctx.workspace.submissions_dir.glob("*lr_baseline*.csv"))
    assert len(submissions) == 1
    assert "submission" in result.lower() or "wrote" in result.lower()


def test_submit_local_includes_id_column_from_test(comp_ctx):
    ml_h.submit_local(comp_ctx, label="run1")
    sub_path = next(comp_ctx.workspace.submissions_dir.glob("*run1*.csv"))
    sub = pd.read_csv(sub_path)
    # Must have the id column from test.csv plus the target column
    assert "id" in sub.columns
    assert comp_ctx.target_col in sub.columns or "target" in sub.columns
    # Row count matches test set
    assert len(sub) == 50


def test_submit_local_emits_one_column_per_class_for_multiclass_proba(tmp_path):
    """F6: with a multi-class needs_proba metric (logloss), submit_local
    must write one column per class instead of stuffing arrays into one
    cell. Columns: id, target_0, target_1, target_2. Each row sums ~= 1.
    """
    import numpy as np

    ws = Workspace.create(root=tmp_path / "comp")
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 60
    train_df = pd.DataFrame({
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train),
        "target": rng.integers(0, 3, size=n_train),
    })
    train_df.to_csv(ws.raw_dir / "train.csv", index=False)
    test_df = pd.DataFrame({
        "id": range(n_test),
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test),
    })
    test_df.to_csv(ws.raw_dir / "test.csv", index=False)
    _write_stub_fe(ws)
    _write_stub_model(ws)
    ctx = _Ctx(workspace=ws, journal=Journal(ws), metric_name="logloss")

    ml_h.submit_local(ctx, label="mc")
    sub_path = next(ws.submissions_dir.glob("*mc*.csv"))
    sub = pd.read_csv(sub_path)
    assert list(sub.columns) == ["id", "target_0", "target_1", "target_2"]
    assert len(sub) == n_test
    sums = sub[["target_0", "target_1", "target_2"]].sum(axis=1)
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-6)


def test_submit_local_requires_fe_and_model(comp_ctx):
    comp_ctx.workspace.fe_path.unlink()
    with pytest.raises(ToolError, match="fe.py"):
        ml_h.submit_local(comp_ctx, label="x")


def test_detect_id_column_does_not_match_substrings():
    """F9: endswith('id') matched `valid`, `covid`, `paranoid`, etc.
    The new algorithm requires the column name to *equal* 'id', end in
    '_id', or start with 'id_'."""
    df = pd.DataFrame({
        "valid": [0, 1],
        "covid": [0, 1],
        "paranoid": [0, 1],
        "x1": [0.1, 0.2],
    })
    assert ml_h._detect_id_column(df) is None


def test_detect_id_column_matches_suffix():
    """F9: a column with the proper `_id` suffix should still be detected."""
    df = pd.DataFrame({"x1": [0, 1], "entry_id": [10, 20]})
    assert ml_h._detect_id_column(df) == "entry_id"


def test_done_sets_ctx_finished(comp_ctx):
    msg = ml_h.done(comp_ctx, summary="best cv was 0.82 with lr baseline")
    assert "0.82" in msg
    assert comp_ctx.finished is True
    assert comp_ctx.final_summary == "best cv was 0.82 with lr baseline"
