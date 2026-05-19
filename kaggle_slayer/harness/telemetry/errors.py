"""Error capture — JSON crash reports for unhandled exceptions.

CLI's outer try/except calls `capture(exc, recent_calls, env)` and we
dump `<ts>_<exctype>.json` to ~/.kaggle_slayer/errors/, capturing the
traceback, the last N tool calls (for context), and a redacted snapshot
of the environment. Rotation: keep the last 100 reports.

Redaction rule: any env key whose UPPERCASE name contains KEY, TOKEN,
SECRET, PASSWORD, PASSWD, AUTH, BEARER, COOKIE, CREDENTIAL, PAT, or
PRIVATE has its value replaced with "<redacted>". This is a coarse
filter — the dev should still review reports before sharing.

The module is single-process / single-threaded; concurrent capture
calls might interleave the file list during rotation but never lose
data (each crash report is its own file).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import traceback
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path.home() / ".kaggle_slayer" / "errors"
_MAX_FILES = 100
# Most patterns are broad substring matches. PAT is special: as a bare
# substring it would catch PATH, PATTERN, PATCH, etc. — so we require it
# to sit at a token boundary (start/end of string or adjacent to `_`).
_REDACT_RE = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|BEARER|COOKIE|CREDENTIAL|PRIVATE)"
    r"|(?:^|_)PAT(?:_|$)"
)


def _now_filename(exc: BaseException) -> str:
    # Microsecond resolution prevents filename collisions when two crashes
    # occur in the same UTC second (e.g., a cascade); without %f the second
    # write would silently overwrite the first.
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H%M%S_%f")
    safe_type = type(exc).__name__.replace(".", "_")
    return f"{stamp}_{safe_type}.json"


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in env.items():
        if _REDACT_RE.search(k.upper()):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _prune(directory: Path) -> None:
    files = sorted(directory.glob("*.json"))
    if len(files) > _MAX_FILES:
        for f in files[: len(files) - _MAX_FILES]:
            with contextlib.suppress(OSError):
                f.unlink()  # best-effort


def capture(
    exc: BaseException,
    *,
    recent_calls: list[dict[str, Any]],
    env: dict[str, str],
    directory: Path | None = None,
) -> Path:
    """Write one crash report and prune older ones to _MAX_FILES."""
    d = Path(directory) if directory is not None else DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _now_filename(exc)
    record = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "recent_calls": recent_calls,
        "env": _redact_env(env),
    }
    path.write_text(json.dumps(record, indent=2))
    _prune(d)
    return path
