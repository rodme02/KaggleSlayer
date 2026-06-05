"""Tests for kaggle_slayer.harness.data.ensure_competition_data."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kaggle_slayer.harness.data import (
    DownloadError,
    ensure_competition_data,
)
from kaggle_slayer.harness.workspace import Workspace


def _make_workspace(tmp_path) -> Workspace:
    return Workspace.create(root=tmp_path / "comp")


def _write_zip(path: Path, **csvs: pd.DataFrame) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, df in csvs.items():
            zf.writestr(name, df.to_csv(index=False))


def test_downloads_and_extracts_when_raw_empty(tmp_path):
    ws = _make_workspace(tmp_path)

    def fake_download(name, *, dest):
        _write_zip(
            Path(dest) / f"{name}.zip",
            **{
                "train.csv": pd.DataFrame({"x": [1, 2], "y": [0, 1]}),
                "test.csv": pd.DataFrame({"x": [3]}),
            },
        )
        return Path(dest)

    client = MagicMock()
    client.download.side_effect = fake_download

    result = ensure_competition_data(ws, client, slug="titanic")

    client.download.assert_called_once_with("titanic", dest=ws.raw_dir)
    assert result.downloaded is True
    assert result.files == ["test.csv", "train.csv"]
    assert (ws.raw_dir / "train.csv").exists()
    # The downloaded zip is removed after extraction.
    assert list(ws.raw_dir.glob("*.zip")) == []


def test_handles_plain_csv_download_no_zip(tmp_path):
    ws = _make_workspace(tmp_path)

    def fake_download(name, *, dest):
        pd.DataFrame({"x": [1], "y": [0]}).to_csv(Path(dest) / "train.csv", index=False)
        return Path(dest)

    client = MagicMock()
    client.download.side_effect = fake_download

    result = ensure_competition_data(ws, client, slug="x")

    assert result.downloaded is True
    assert result.files == ["train.csv"]


def test_skips_when_csv_present(tmp_path):
    ws = _make_workspace(tmp_path)
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(ws.raw_dir / "train.csv", index=False)

    client = MagicMock()
    result = ensure_competition_data(ws, client, slug="titanic")

    client.download.assert_not_called()
    assert result.downloaded is False
    assert result.files == ["train.csv"]


def test_disabled_never_calls_client(tmp_path):
    ws = _make_workspace(tmp_path)
    client = MagicMock()

    result = ensure_competition_data(ws, client, slug="titanic", enabled=False)

    client.download.assert_not_called()
    assert result.downloaded is False
    assert result.files == []


def test_client_failure_raises_download_error(tmp_path):
    ws = _make_workspace(tmp_path)
    client = MagicMock()
    client.download.side_effect = RuntimeError("403 Forbidden")

    with pytest.raises(DownloadError) as ex:
        ensure_competition_data(ws, client, slug="titanic")

    assert ex.value.slug == "titanic"
    assert "403" in str(ex.value.cause)


def test_corrupt_zip_raises_download_error(tmp_path):
    ws = _make_workspace(tmp_path)

    def fake_download(name, *, dest):
        # Write a file with a .zip name that is not a valid zip archive.
        (Path(dest) / f"{name}.zip").write_bytes(b"not a real zip")
        return Path(dest)

    client = MagicMock()
    client.download.side_effect = fake_download

    with pytest.raises(DownloadError) as ex:
        ensure_competition_data(ws, client, slug="titanic")

    assert ex.value.slug == "titanic"


def test_nested_csv_not_reported_as_top_level(tmp_path):
    """A CSV that extracts into a subdirectory is not usable top-level data.

    Downstream consumers (context.py, handlers/ml.py) read raw/train.csv at
    the top level, so _existing_csvs counts only top-level CSVs. A nested
    layout extracts but yields no usable files — a known, out-of-scope gap
    we pin here so it isn't silently "fixed" into a false success.
    """
    ws = _make_workspace(tmp_path)

    def fake_download(name, *, dest):
        _write_zip(
            Path(dest) / f"{name}.zip",
            **{f"{name}/train.csv": pd.DataFrame({"x": [1], "y": [0]})},
        )
        return Path(dest)

    client = MagicMock()
    client.download.side_effect = fake_download

    result = ensure_competition_data(ws, client, slug="housing")

    # The archive was extracted to the subdirectory...
    assert (ws.raw_dir / "housing" / "train.csv").exists()
    # ...but a nested CSV is not counted as usable top-level data.
    assert result.downloaded is True
    assert result.files == []
