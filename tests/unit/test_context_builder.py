"""Tests for kaggle_slayer.harness.context.build_context."""

from __future__ import annotations

import pandas as pd
import pytest

from kaggle_slayer.harness import context as ctx_mod
from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace


class FakeKaggleClient:
    def __init__(
        self,
        info: CompetitionInfo,
        files: list[CompetitionFile],
        lb: list[LBEntry],
    ):
        self.info = info
        self.files = files
        self.lb = lb

    def view_competition(self, name: str) -> CompetitionInfo:
        return self.info

    def list_files(self, name: str) -> list[CompetitionFile]:
        return self.files

    def get_leaderboard(self, name: str, *, top_n: int = 10) -> list[LBEntry]:
        return self.lb[:top_n]


@pytest.fixture
def workspace(tmp_path):
    return Workspace.create(root=tmp_path / "titanic")


@pytest.fixture
def kaggle_fake():
    return FakeKaggleClient(
        info=CompetitionInfo(
            title="Titanic - Machine Learning from Disaster",
            description="Predict survival on the Titanic.",
            metric="accuracy",
        ),
        files=[
            CompetitionFile(name="train.csv", size=60302),
            CompetitionFile(name="test.csv", size=28629),
            CompetitionFile(name="gender_submission.csv", size=3258),
        ],
        lb=[
            LBEntry(team_name="alpha", score=1.0),
            LBEntry(team_name="beta", score=0.99999),
        ],
    )


@pytest.fixture
def sample_train_csv(workspace):
    df = pd.DataFrame({
        "PassengerId": range(1, 11),
        "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
    })
    train_path = workspace.raw_dir / "train.csv"
    df.to_csv(train_path, index=False)
    return train_path


def test_build_context_writes_context_md(workspace, kaggle_fake, sample_train_csv):
    path = ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    assert path == workspace.context_path
    assert path.exists()


def test_build_context_includes_title_and_metric(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "Titanic" in body
    assert "accuracy" in body


def test_build_context_lists_data_files(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "train.csv" in body
    assert "test.csv" in body


def test_build_context_includes_leaderboard_summary(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    # Top score is shown, even if team name redacted
    assert "1.0" in body or "1.00000" in body


def test_build_context_profiles_train_data(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    # Columns appear in the data summary
    assert "Survived" in body
    assert "Age" in body
    # Row count
    assert "10" in body


def test_build_context_suggests_target_column(workspace, kaggle_fake, sample_train_csv):
    """A column named 'Survived' or 'target' should be flagged as the likely target."""
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "target" in body
    assert "survived" in body


def test_build_context_works_without_train_csv(workspace, kaggle_fake):
    """If raw/train.csv doesn't exist, the data section says so but the file is still written."""
    path = ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = path.read_text()
    assert "no train.csv" in body.lower() or "train data not yet downloaded" in body.lower()


def test_build_context_overwrites_existing(workspace, kaggle_fake, sample_train_csv):
    workspace.context_path.write_text("stale content")
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "stale content" not in body
    assert "Titanic" in body


def test_target_hint_matches_case_insensitively(workspace, kaggle_fake, tmp_path):
    """A column 'TARGET' or 'Target' should be flagged the same as 'target'."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "Target": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "likely target" in body
    assert "target" in body  # the column name appears as a flagged candidate


def test_target_hint_matches_suffix_label(workspace, kaggle_fake):
    """A column ending in '_label' or '_target' should be flagged."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "class_label": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "likely target" in body
    assert "class_label" in body


def test_target_hint_no_match_says_so(workspace, kaggle_fake):
    """If no column matches the hint patterns, context.md should say the agent must infer."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "outcome_var": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "agent should infer" in body
