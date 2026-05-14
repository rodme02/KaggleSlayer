"""Wrapper around the kaggle library (v2.1+ structured-response shape).

Provides typed return values (CompetitionInfo, CompetitionFile, LBEntry)
so the rest of the harness doesn't depend on kaggle's internal dataclasses.

The kaggle library authenticates on first use from KAGGLE_API_TOKEN or
~/.kaggle/access_token (new format) or KAGGLE_USERNAME+KAGGLE_KEY or
~/.kaggle/kaggle.json (legacy). The wrapper does not duplicate that
logic; if no creds are present, the underlying library raises and we
propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionInfo:
    title: str
    description: str
    metric: str | None


@dataclass(frozen=True)
class CompetitionFile:
    name: str
    size: int


@dataclass(frozen=True)
class LBEntry:
    team_name: str
    score: float


def _get_api() -> Any:  # noqa: ANN401 — kaggle library is untyped
    """Lazy import + authenticate; kept as a function so tests can patch it."""
    from kaggle import api  # type: ignore[import-untyped]

    api.authenticate()
    return api


def _safe_attr(obj: object, name: str, default: object = None) -> Any:  # noqa: ANN401
    return getattr(obj, name, default)


class KaggleClient:
    """Read-only-by-default wrapper. submit() is the one write op."""

    def view_competition(self, name: str) -> CompetitionInfo:
        api = _get_api()
        resp = api.competition_view(name)
        return CompetitionInfo(
            title=_safe_attr(resp, "title", ""),
            description=_safe_attr(resp, "description", ""),
            metric=_safe_attr(resp, "evaluation_metric"),
        )

    def list_files(self, name: str) -> list[CompetitionFile]:
        api = _get_api()
        resp = api.competition_list_files(name)
        files = _safe_attr(resp, "files", []) or []
        return [
            CompetitionFile(
                name=_safe_attr(f, "name", ""),
                size=int(_safe_attr(f, "size", 0) or 0),
            )
            for f in files
        ]

    def download(self, name: str, *, dest: Path) -> Path:
        """Download all competition files into `dest`. Returns dest."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        api = _get_api()
        api.competition_download_files(name, path=str(dest))
        return dest

    def get_leaderboard(self, name: str, *, top_n: int = 50) -> list[LBEntry]:
        api = _get_api()
        resp = api.competition_view_leaderboard(name)
        entries = _safe_attr(resp, "submissions", []) or []
        result: list[LBEntry] = []
        for entry in entries[:top_n]:
            raw_score = _safe_attr(entry, "score", "0")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            result.append(LBEntry(team_name=_safe_attr(entry, "team_name", ""), score=score))
        return result

    def submit(self, name: str, *, csv_path: Path, message: str) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"submission CSV not found: {csv_path}")
        api = _get_api()
        api.competition_submit(str(csv_path), message, name)
