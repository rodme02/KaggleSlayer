# Auto-Download Competition Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kaggle-slayer competitions/<name>` self-fetch the competition data into `raw/` when it's missing, instead of requiring the user to populate `raw/` by hand.

**Architecture:** A new `kaggle_slayer/harness/data.py` owns the "is data already here? → download → unzip → clean up" orchestration. It calls only `KaggleClient.download` (the sole `kaggle.api` boundary, hard rule #4) plus stdlib `zipfile`. `cli.py` calls it once in `_run_inner`, before `build_context`, and hard-exits (code 2) with an actionable message when a needed download fails.

**Tech Stack:** Python 3, stdlib `zipfile`, `dataclasses`, `typing.Protocol`; pytest + `unittest.mock` for tests (no live Kaggle).

**Reference spec:** `docs/superpowers/specs/2026-06-04-auto-download-competition-data-design.md`

---

## File Structure

- **Create** `kaggle_slayer/harness/data.py` — `DownloadResult`, `DownloadError`, `ensure_competition_data`, plus private `_extract_zips` / `_existing_csvs`. One responsibility: ensure `raw/` is populated.
- **Create** `tests/unit/test_data.py` — unit tests for the module with a stub client.
- **Modify** `kaggle_slayer/cli.py` — add `--no-download` / `--competition` flags, import + call `ensure_competition_data`, add `_download_error_message` helper.
- **Modify** `tests/unit/test_cli.py` — flag-parsing + behavior tests.
- **Modify** `README.md` — quickstart text + flags table.
- **Modify** `CLAUDE.md` — layout map entry for `harness/data.py`.

Existing CLI tests pre-populate `raw/train.csv`, so the skip-if-CSV-present guard keeps them green with no changes.

---

## Task 1: `harness/data.py` — the orchestration module

**Files:**
- Create: `kaggle_slayer/harness/data.py`
- Test: `tests/unit/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_data.py`:

```python
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


def test_client_failure_raises_download_error(tmp_path):
    ws = _make_workspace(tmp_path)
    client = MagicMock()
    client.download.side_effect = RuntimeError("403 Forbidden")

    with pytest.raises(DownloadError) as ex:
        ensure_competition_data(ws, client, slug="titanic")

    assert ex.value.slug == "titanic"
    assert "403" in str(ex.value.cause)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_slayer.harness.data'` (collection error).

- [ ] **Step 3: Write the implementation**

Create `kaggle_slayer/harness/data.py`:

```python
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

    _extract_zips(raw_dir)

    return DownloadResult(slug=slug, downloaded=True, files=_existing_csvs(raw_dir))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_data.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint + type-check the new module**

Run: `ruff check kaggle_slayer/harness/data.py tests/unit/test_data.py && mypy kaggle_slayer/harness`
Expected: no errors (mypy is strict on `harness/` and must stay clean).

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/data.py tests/unit/test_data.py
git commit -m "feat(harness): ensure_competition_data — auto-fetch + unzip into raw/

New harness/data.py orchestrates the one-time download: skip when raw/
already has CSVs, else KaggleClient.download + unzip + clean up. Wraps
client failures in a typed DownloadError. Kaggle API access stays in
KaggleClient (hard rule #4).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire auto-download into the CLI

**Files:**
- Modify: `kaggle_slayer/cli.py` (imports near line 33; `_parse_args` ~line 60; `_run_inner` ~line 137)
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_parses_download_flags(tmp_path):
    args = cli._parse_args([
        str(tmp_path / "comp"), "--target", "y",
        "--no-download", "--competition", "titanic",
    ])
    assert args.no_download is True
    assert args.competition == "titanic"


def test_cli_download_flags_default(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--target", "y"])
    assert args.no_download is False
    assert args.competition is None


def test_cli_no_download_passes_enabled_false(tmp_path):
    """--no-download must reach ensure_competition_data as enabled=False."""
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    raw = comp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(raw / "test.csv", index=False)

    with patch("kaggle_slayer.cli.ensure_competition_data") as mock_ensure, \
         patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--no-download"])

    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.kwargs["enabled"] is False


def test_cli_download_failure_exits_2(tmp_path):
    """A DownloadError from a needed fetch hard-exits with code 2."""
    comp_path = tmp_path / "comp"

    with patch(
        "kaggle_slayer.cli.ensure_competition_data",
        side_effect=cli.DownloadError("titanic", RuntimeError("403 Forbidden")),
    ), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        exit_code = cli.run([str(comp_path), "--target", "y"])

    assert exit_code == 2
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/unit/test_cli.py -k "download" -v`
Expected: FAIL — `AttributeError: <module 'kaggle_slayer.cli'> does not have the attribute 'ensure_competition_data'` (and the flag-parse tests fail on unknown `--no-download`).

- [ ] **Step 3: Add the import**

In `kaggle_slayer/cli.py`, add after the existing `from kaggle_slayer.harness.kaggle_client import KaggleClient` line (~line 33):

```python
from kaggle_slayer.harness.data import DownloadError, ensure_competition_data
```

- [ ] **Step 4: Add the CLI flags**

In `_parse_args`, after the `--no-context-build` argument (~line 61), add:

```python
    p.add_argument("--no-download", action="store_true",
                   help="Skip auto-downloading competition data into raw/ (use existing/manual data)")
    p.add_argument("--competition", default=None,
                   help="Kaggle competition slug to download (defaults to the workspace dir name)")
```

- [ ] **Step 5: Add the error-message helper**

In `kaggle_slayer/cli.py`, add this module-level function just above `def run(` (~line 90):

```python
def _download_error_message(e: DownloadError) -> str:
    """Render an actionable message for a failed competition download."""
    cause = str(e.cause).lower()
    base = f"ERROR: could not download competition {e.slug!r}"
    tail = "Or pass --no-download if you'll provide data in raw/ yourself."
    if "403" in cause or "forbidden" in cause:
        return (
            f"{base}: access denied. Accept the competition rules at "
            f"https://www.kaggle.com/c/{e.slug}/rules, then rerun. {tail}"
        )
    if any(s in cause for s in ("kaggle.json", "credential", "authenticate", "401", "unauthorized")):
        return (
            f"{base}: no working Kaggle credentials. Set KAGGLE_API_TOKEN "
            f"or run `python scripts/preflight.py`. {tail}"
        )
    return f"{base}: {e.cause}. {tail}"
```

- [ ] **Step 6: Call `ensure_competition_data` in `_run_inner`**

In `_run_inner`, immediately after the API-key check block that ends with `return 2` (~line 137) and **before** the `# Build context.md ...` block (~line 139), insert:

```python
    # Auto-fetch competition data into raw/ unless opted out. Runs before
    # build_context, which reads raw/train.csv for the data brief. A needed
    # download that fails hard-exits (code 2) rather than letting the Solver
    # burn tokens on empty data; --no-download is the escape hatch.
    slug = args.competition or workspace.name
    try:
        ensure_competition_data(
            workspace, KaggleClient(), slug=slug, enabled=not args.no_download,
        )
    except DownloadError as e:
        print(_download_error_message(e), file=sys.stderr)
        return 2
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `pytest tests/unit/test_cli.py -k "download" -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Run the full unit suite + lint + type-check**

Run: `pytest -m "not slow" -q && ruff check kaggle_slayer tests && mypy kaggle_slayer/harness kaggle_slayer/agent`
Expected: all pass; existing CLI tests stay green (they pre-populate `raw/train.csv`, so `ensure_competition_data` skips without calling the client).

- [ ] **Step 9: Commit**

```bash
git add kaggle_slayer/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): auto-download competition data before solve

kaggle-slayer now fetches into raw/ when it has no CSVs. Adds
--no-download / --competition flags; a needed-but-failed download exits
2 with an actionable message (rules / credentials / underlying error).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Documentation

**Files:**
- Modify: `README.md` (quickstart block + flags table)
- Modify: `CLAUDE.md` (layout map)

- [ ] **Step 1: Update the README quickstart**

In `README.md`, replace this line in the quickstart code block:

```bash
# competitions/<name>/raw/ should contain Kaggle's train.csv + test.csv
kaggle-slayer competitions/titanic --target Survived --metric accuracy
```

with:

```bash
# kaggle-slayer auto-downloads competition data into raw/ on first run
# (the slug is the workspace dir name; pass --no-download to use your own data)
kaggle-slayer competitions/titanic --target Survived --metric accuracy
```

- [ ] **Step 2: Add the two flags to the README flags table**

In `README.md`, after the `| `--no-context-build` | Skip context rebuild entirely (manual `context.md`). |` row, add:

```markdown
| `--no-download` | Skip auto-downloading competition data into `raw/` (use data you placed there yourself). |
| `--competition <slug>` | Kaggle competition slug to download. Defaults to the workspace directory name. |
```

- [ ] **Step 3: Add `data.py` to the CLAUDE.md layout map**

In `CLAUDE.md`, in the `harness/` section of the layout tree, add a line after the `context.py` entry:

```
│   ├─ data.py                     # ensure_competition_data — auto-fetch + unzip into raw/
```

- [ ] **Step 4: Sanity-check the docs render**

Run: `git diff --stat README.md CLAUDE.md`
Expected: both files show as modified; eyeball the diff for correct markdown table/tree alignment.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document auto-download + --no-download/--competition flags

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the full non-slow suite once more**

Run: `pytest -m "not slow" -q`
Expected: all pass (prior count + 9 new tests: 5 in test_data.py, 4 in test_cli.py).

- [ ] **Confirm clean tree**

Run: `git status`
Expected: clean working tree; three new commits on `chore/v1-release-prep`.

---

## Notes for the implementer

- **Hard rule #4:** never call `kaggle.api.*` from `data.py` — go through `KaggleClient.download`. The only Kaggle access added here is that one call.
- **Why exit-2 (not warn-and-continue):** unlike `build_context` (which degrades gracefully to a thinner brief), an empty `raw/` means the Solver has nothing to work with — failing fast avoids wasted Gemini spend. This is the one deliberate behavioral departure; it's covered by `test_cli_download_failure_exits_2`.
- **No-key/synthetic path stays intact:** auto-download is skipped whenever `raw/` already has a CSV or `--no-download` is passed, so the future credential-free demo is unaffected. Do not add an unconditional download.
- **Out of scope (do not add):** force/refresh re-download flag, nested-zip recursion, non-`train.csv` filename handling, a standalone `kaggle-slayer-download` command.
```