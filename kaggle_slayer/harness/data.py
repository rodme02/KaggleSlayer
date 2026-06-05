"""Ensure a competition's data is present in the workspace's raw/ dir.

Orchestrates the one-time fetch: if raw/ has no CSVs, download the
competition via KaggleClient and unzip the archive(s) in place. The only
Kaggle-API access stays inside KaggleClient (hard rule #4); this module
adds the pure-Python "is it already here? -> download -> unzip" glue so
the CLI stays thin and the logic is testable without a live Kaggle login.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kaggle_slayer.harness.workspace import Workspace

logger = logging.getLogger(__name__)


class _KaggleClientLike(Protocol):
    """Structural type for the bit of KaggleClient this module needs."""

    def download(self, name: str, *, dest: Path) -> Path: ...


@dataclass(frozen=True)
class DownloadResult:
    slug: str
    downloaded: bool  # False when skipped because data was already present
    files: list[str]  # CSV file names now in raw/ (sorted)


class DownloadError(Exception):
    """Raised when a *needed* competition download fails.

    Carries the slug and the underlying cause so the CLI can render an
    actionable message (rules not accepted, missing credentials, ...).
    """

    def __init__(self, slug: str, cause: Exception) -> None:
        self.slug = slug
        self.cause = cause
        super().__init__(f"failed to download competition {slug!r}: {cause}")


def _existing_csvs(raw_dir: Path) -> list[str]:
    # Top-level only: every consumer (context.py, handlers/ml.py) reads
    # raw/<name>.csv at the top of raw/, so a CSV nested in a subdirectory
    # is not usable data and must not satisfy the skip-guard.
    return sorted(p.name for p in raw_dir.glob("*.csv"))


def _extract_zips(raw_dir: Path) -> None:
    """Extract every top-level *.zip in raw/ and delete it afterwards."""
    for zip_path in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()


def ensure_competition_data(
    workspace: Workspace,
    kaggle_client: _KaggleClientLike,
    *,
    slug: str,
    enabled: bool = True,
) -> DownloadResult:
    """Make sure raw/ has the competition data; download + unzip if missing.

    Skips (no network) when disabled or when raw/ already holds any CSV.
    Raises DownloadError if a needed download fails.
    """
    raw_dir = workspace.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_csvs(raw_dir)
    if not enabled or existing:
        if existing:
            logger.debug("raw/ already has %d csv(s); skipping download", len(existing))
        return DownloadResult(slug=slug, downloaded=False, files=existing)

    try:
        kaggle_client.download(slug, dest=raw_dir)
    except Exception as e:  # noqa: BLE001 — wrap any client/auth/network failure
        raise DownloadError(slug, e) from e

    try:
        _extract_zips(raw_dir)
    except Exception as e:  # noqa: BLE001 — a corrupt archive is a failed download
        raise DownloadError(slug, e) from e

    return DownloadResult(slug=slug, downloaded=True, files=_existing_csvs(raw_dir))
