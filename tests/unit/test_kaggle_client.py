"""Tests for kaggle_slayer.harness.kaggle_client.

Most tests mock the underlying kaggle library; live-API tests live in
tests/integration/test_real_apis.py (slow tier, skipped by default).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kaggle_slayer.harness import kaggle_client as kc_mod


@pytest.fixture
def mock_api(monkeypatch):
    """Patch kaggle_client._get_api() to return a MagicMock."""
    api = MagicMock(name="kaggle_api")
    monkeypatch.setattr(kc_mod, "_get_api", lambda: api)
    return api


def test_client_view_competition(mock_api):
    comp = MagicMock()
    comp.title = "Titanic - Machine Learning from Disaster"
    comp.description = "Predict survival on the Titanic..."
    comp.evaluation_metric = "Categorization Accuracy"
    comp.ref = "https://www.kaggle.com/competitions/titanic"

    other = MagicMock()
    other.title = "Other Comp"
    other.description = "..."
    other.evaluation_metric = "rmse"
    other.ref = "https://www.kaggle.com/competitions/other"

    resp = MagicMock()
    resp.competitions = [other, comp]
    mock_api.competitions_list.return_value = resp

    client = kc_mod.KaggleClient()
    info = client.view_competition("titanic")

    # competitions_list should be called with the search kwarg
    mock_api.competitions_list.assert_called_once_with(search="titanic")
    assert info.title.startswith("Titanic")
    assert info.description.startswith("Predict")
    assert info.metric == "Categorization Accuracy"


def test_client_view_competition_handles_missing_metric(mock_api):
    comp = MagicMock(spec=["title", "description", "ref"])
    comp.title = "Untitled Comp"
    comp.description = "..."
    comp.ref = "https://www.kaggle.com/competitions/foo"
    resp = MagicMock()
    resp.competitions = [comp]
    mock_api.competitions_list.return_value = resp

    client = kc_mod.KaggleClient()
    info = client.view_competition("foo")
    assert info.metric is None


def test_client_view_competition_no_match_raises(mock_api):
    """If no comp's ref matches the requested name, raise a clear error."""
    other = MagicMock()
    other.ref = "https://www.kaggle.com/competitions/different"
    resp = MagicMock()
    resp.competitions = [other]
    mock_api.competitions_list.return_value = resp

    client = kc_mod.KaggleClient()
    with pytest.raises(LookupError, match="titanic"):
        client.view_competition("titanic")


def test_client_view_competition_empty_results_raises(mock_api):
    resp = MagicMock()
    resp.competitions = []
    mock_api.competitions_list.return_value = resp
    client = kc_mod.KaggleClient()
    with pytest.raises(LookupError):
        client.view_competition("nonexistent")


def test_client_list_files(mock_api):
    file_a = MagicMock()
    file_a.name = "train.csv"
    file_a.size = 60302
    file_b = MagicMock()
    file_b.name = "test.csv"
    file_b.size = 28629
    resp = MagicMock()
    resp.files = [file_a, file_b]
    mock_api.competition_list_files.return_value = resp

    client = kc_mod.KaggleClient()
    files = client.list_files("titanic")
    assert [f.name for f in files] == ["train.csv", "test.csv"]
    assert files[0].size == 60302


def test_client_get_leaderboard(mock_api):
    e1 = MagicMock()
    e1.team_name = "team_a"
    e1.score = "1.00000"
    e2 = MagicMock()
    e2.team_name = "team_b"
    e2.score = "0.99999"
    resp = MagicMock()
    resp.submissions = [e1, e2]
    mock_api.competition_view_leaderboard.return_value = resp

    client = kc_mod.KaggleClient()
    lb = client.get_leaderboard("titanic", top_n=2)
    assert len(lb) == 2
    assert lb[0].team_name == "team_a"
    assert lb[0].score == 1.0


def test_client_get_leaderboard_truncates_to_top_n(mock_api):
    entries = []
    for i in range(20):
        e = MagicMock()
        e.team_name = f"team_{i}"
        e.score = f"{1.0 - i * 0.01:.5f}"
        entries.append(e)
    resp = MagicMock()
    resp.submissions = entries
    mock_api.competition_view_leaderboard.return_value = resp

    client = kc_mod.KaggleClient()
    lb = client.get_leaderboard("titanic", top_n=5)
    assert len(lb) == 5
    assert lb[0].team_name == "team_0"


def test_client_download_returns_target_dir(mock_api, tmp_path):
    target = tmp_path / "raw"
    client = kc_mod.KaggleClient()
    result = client.download("titanic", dest=target)
    mock_api.competition_download_files.assert_called_once()
    args, kwargs = mock_api.competition_download_files.call_args
    assert args[0] == "titanic" or kwargs.get("competition") == "titanic"
    assert result == target


def test_client_submit(mock_api, tmp_path):
    csv = tmp_path / "submission.csv"
    csv.write_text("id,target\n1,0\n")
    mock_api.competition_submit.return_value = MagicMock(spec=[])

    client = kc_mod.KaggleClient()
    client.submit("titanic", csv_path=csv, message="cv=0.842")
    mock_api.competition_submit.assert_called_once()


def test_client_submit_rejects_missing_csv(mock_api, tmp_path):
    client = kc_mod.KaggleClient()
    with pytest.raises(FileNotFoundError):
        client.submit("titanic", csv_path=tmp_path / "nope.csv", message="x")
