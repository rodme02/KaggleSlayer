# Auto-Download Competition Data — Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-06-04
**Author:** Rodrigo Medeiros (with Claude)
**Relates to:** `2026-05-14-llm-agent-harness-design.md` (§4 hard rules, §10 workspace layout)

---

## 1. Goal

Make running a competition self-contained: `kaggle-slayer competitions/<name> --target <col>`
should fetch the competition data itself when it's missing, instead of requiring the user
to manually populate `raw/` with `train.csv` / `test.csv` out of band.

Today there is no download UX. `KaggleClient.download()` exists but is wired to nothing;
`build_context` only *lists* files for the brief and assumes `raw/train.csv` is already
present (`context.py` prints *"Train data not yet downloaded"* otherwise). The README
tells users to populate `raw/` by hand.

## 2. Scope

**In scope:**
- Auto-download competition data into `<workspace>/raw/` during the solve flow, before
  context build, when `raw/` has no CSV files.
- Unzip the downloaded archive(s) and remove them.
- Two new CLI flags: `--no-download` and `--competition <slug>`.

**Out of scope (YAGNI):**
- A force/refresh re-download flag — to refetch, delete `raw/` and rerun.
- Nested-zip recursion (zip-within-zip). One level of extraction is enough for tabular
  Playground comps.
- Non-`train.csv` file-name conventions — an existing limitation in `context.py`, not
  introduced or fixed here.
- A standalone `kaggle-slayer-download` command or a `download` subcommand. The chosen UX
  is auto-download integrated into the solve run.

## 3. Behavior

### 3.1 Trigger

In `_run_inner`, **before** `build_context` (which reads `raw/train.csv`):

1. If `--no-download` is set → skip entirely; never touch the network.
2. Else compute the Kaggle slug: `--competition <slug>` if given, otherwise the workspace
   directory name (`workspace.name` — the same mapping `build_context` already uses).
3. If `raw/` already contains at least one `*.csv` → skip (log at debug). This keeps
   manually-prepared workspaces and the future no-key/synthetic-comp path untouched.
4. Otherwise download the competition into `raw/`, extract every top-level `*.zip`,
   delete the zip(s), and continue into context-build → solve.

The "has CSV" sentinel (`any(raw_dir.glob("*.csv"))`) is deliberately broader than checking
for `train.csv` specifically, so a comp whose files are already present in any form does not
re-trigger a download.

### 3.2 Error handling

- **Data already present** → silent skip (debug log).
- **Download needed but fails** → hard exit, **return code 2**, with an actionable message
  that names the likely cause:
  - HTTP 403 / forbidden → "accept the competition rules at
    `https://www.kaggle.com/c/<slug>/rules`, then rerun".
  - Missing credentials → "no Kaggle credentials found — set `KAGGLE_API_TOKEN` or run
    `python scripts/preflight.py`".
  - Other (network, unknown slug) → surface the underlying error message.

  Rationale: failing fast beats burning Gemini tokens solving on empty data. `--no-download`
  is the documented escape hatch for "I'll provide data myself".
- This is a deliberate departure from `build_context`'s non-fatal warn-and-continue pattern:
  context build can degrade gracefully (the agent can run with a thinner brief), but *no data
  at all* cannot.

## 4. Components

### 4.1 `KaggleClient.download()` — unchanged responsibility

Stays the thin wrapper over `api.competition_download_files`. Per hard rule #4, this is the
only place allowed to touch `kaggle.api.*`. It already creates `dest`, downloads, and returns
the path. No behavior change; unzip does **not** move here (unzip needs no Kaggle API, so it
stays out of the API-boundary module and remains testable without mocking the API).

### 4.2 New module `kaggle_slayer/harness/data.py`

Owns the orchestration. Single public function plus a small typed result:

```python
@dataclass(frozen=True)
class DownloadResult:
    slug: str
    downloaded: bool          # False if skipped because data already present
    files: list[str]          # CSV file names now in raw/ (sorted)

def ensure_competition_data(
    workspace: Workspace,
    kaggle_client: _KaggleClientLike,   # structural: .download(name, *, dest) -> Path
    *,
    slug: str,
    enabled: bool = True,
) -> DownloadResult:
    ...
```

Responsibilities, in order:
1. If `not enabled` → return `DownloadResult(slug, downloaded=False, files=<existing csvs>)`.
2. If `raw/` already has a `*.csv` → return early, `downloaded=False`.
3. `kaggle_client.download(slug, dest=workspace.raw_dir)`.
4. Extract every top-level `*.zip` in `raw/` with `zipfile`, then unlink the zip(s).
5. Return `DownloadResult(slug, downloaded=True, files=<csvs now in raw/>)`.

Errors from the client (no creds / 403 / unknown comp) propagate as a typed
`DownloadError(slug, cause)` so the CLI can render the actionable message and exit 2.
`_KaggleClientLike` is a `Protocol` (mirrors `context.py`'s `_KaggleClientLike`) so tests
pass a stub.

### 4.3 CLI wiring (`cli.py`)

- Add `--no-download` (`action="store_true"`) and `--competition` (`default=None`) to
  `_parse_args`.
- In `_run_inner`, immediately after `Workspace.create(...)` and the API-key check, and
  **before** the `build_context` block:

  ```python
  slug = args.competition or workspace.name
  try:
      ensure_competition_data(
          workspace, KaggleClient(), slug=slug, enabled=not args.no_download,
      )
  except DownloadError as e:
      print(actionable_message(e), file=sys.stderr)
      return 2
  ```

  cli.py stays thin — one call plus the error-to-message mapping. The same `KaggleClient`
  instance can be reused for `build_context` to avoid re-authenticating.

## 5. Testing

Unit tests for `ensure_competition_data` (stub client, no real Kaggle):
- Downloads + extracts + lists CSVs when `raw/` has no CSV.
- Skips (no client call) when a CSV already exists.
- `enabled=False` never calls the client and never touches the network.
- The downloaded `*.zip` is extracted and then removed.
- A client failure surfaces as `DownloadError`.

`test_cli.py`:
- `--no-download` short-circuits (stubbed client's `download` never called).
- A `DownloadError` path returns exit code 2 with a message naming the slug.

No real-Kaggle calls; everything runs in the `not slow` tier.

## 6. Docs

- **README quickstart:** drop the "`competitions/<name>/raw/` should contain Kaggle's
  train.csv + test.csv" manual step; show that `kaggle-slayer` now self-fetches when `raw/`
  is empty. Add `--no-download` and `--competition` to the flags table.
- **CLAUDE.md layout map:** add `harness/data.py` ("ensure_competition_data — auto-fetch +
  unzip into raw/").

## 7. Hard-rule compliance

- **#4 (Kaggle only through the wrapper):** all `kaggle.api.*` access stays in
  `KaggleClient`; `data.py` calls only `KaggleClient.download` + stdlib `zipfile`.
- **#6 (telemetry never crashes the agent):** unaffected — download is a pre-solve setup
  step, not a telemetry surface; its hard-exit happens before the Solver loop starts.
- **No-key/synthetic path:** preserved — auto-download is skipped whenever `raw/` already
  has data or `--no-download` is passed, so the future credential-free demo is unaffected.
