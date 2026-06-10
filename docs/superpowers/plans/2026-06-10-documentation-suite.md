# Documentation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure docs into a layered, drift-resistant suite for Claude Code-driven development: lean-router CLAUDE.md, `docs/architecture.md`, six ADRs, `.claude/` scaffolding, de-drifted README/GOALS.

**Architecture:** Every fact lives in exactly one place — rules in CLAUDE.md, structure in architecture.md, rationale in ADRs, status in GOALS.md; everything else links. `.claude/` turns recurring workflows into executable artifacts. Router is written last so its pointers target files that exist.

**Tech Stack:** Markdown, Claude Code command/agent/settings conventions. No code changes; gates are `pytest -m "not slow"` (408 passed, 1 skipped as of 2026-06-10), `ruff check kaggle_slayer tests`, `mypy kaggle_slayer/harness kaggle_slayer/agent`.

**Spec:** `docs/superpowers/specs/2026-06-10-documentation-suite-design.md`

**Ground truth to use everywhere** (verified 2026-06-10):
- Non-slow tier: **408 passed, 1 skipped (env-gated), 8 deselected slow** in ~5s. Re-verify with `python -m pytest -m "not slow" 2>&1 | tail -1` before writing any count.
- `--target` is now **required** (was optional with silent `'target'` default).
- opentelemetry-api/sdk were **removed** from deps (in-tree JSONL tracer only).
- `harness/data.py` (auto-download) exists; flags `--no-download`, `--competition`.
- CI (`.github/workflows/ci.yml`) runs `pytest -m "not slow"` + ruff + `mypy kaggle_slayer/harness` on Linux 3.11 + 3.12. CI does NOT run a separate `-m integration` job.

---

### Task 1: `docs/architecture.md`

**Files:**
- Create: `docs/architecture.md`

- [ ] **Step 1: Write the document** with exactly these sections (source facts from the named files — read them, do not recall):

1. `## Trust boundary` — harness (`kaggle_slayer/harness/`) is trusted; agent output (`agent/fe.py`, `agent/model.py`, anything the LLM writes) is never trusted: AST-linted before import (`sandbox.lint_module`), executed under POSIX rlimits (`sandbox.run_subprocess`), and fed only fold-local data during fit (`cv.train_cv`). Include the one-paragraph threat model: the adversary is *accidental* agent code (data leakage, file damage, resource exhaustion), not a malicious human.
2. `## Module map` — move the layout tree currently in CLAUDE.md "Layout" verbatim, then annotate each module with one line (responsibility + key types). Include `harness/data.py`.
3. `## Anatomy of a solve` — numbered walkthrough citing real symbols: `cli.run` → `_run_inner` (env check → `ensure_competition_data` → `build_context`) → `Solver.solve` loop (`llm.call` → `_dispatch` → `Journal.log_tool_call`) → `train_cv` (per-fold `fit_feature_transformer`, lint gate) → checkpoint gate (`CheckpointHandler.request`, four `HandlerMode`s) → `submit_kaggle` (leaderboard.jsonl + calibration row). Note exit codes: 0 done, 1 not-done, 2 setup error, 3 resume failure, 4 crash (report written), 130 interrupt.
4. `## On-disk layout` — per-workspace tree (`raw/`, `context.md`, `agent/` + `agent/versions/`, `submissions/` + `leaderboard.jsonl`, `run_log.jsonl`, `notes.jsonl`, `otel.jsonl`, `mlflow_errors.log`) and global `~/.kaggle_slayer/` (`cost_ledger.jsonl`, `calibration.jsonl`, `errors/`, `mlruns/`).
5. `## Telemetry surfaces` — OTel JSONL tracer (in-tree, single-process), MLflow (file store default), calibration log, crash reports (redaction + 100-file rotation), behavior metrics; state the never-crash contract and where failures are logged.
6. `## Test architecture` — four tiers (unit / integration / chaos / slow), what each may touch (slow = real Gemini, opt-in `-m slow`; chaos = injected 5% LLM failures; unit+integration = no network), fixtures (`fake_llm.py`, `synthetic_comp.py`).
7. `## Files that pin the architecture` — move the list from CLAUDE.md verbatim (cv.py, checkpoints.py, solver.py, llm_client.py, otel.py, mlflow_logger.py) with its read-before-touching framing.

- [ ] **Step 2: Verify** — every file path mentioned exists: `grep -oE 'kaggle_slayer/[a-z_/]+\.py' docs/architecture.md | sort -u | xargs ls` (no errors expected).

- [ ] **Step 3: Commit** — `git add docs/architecture.md && git commit -m "docs: architecture.md — trust boundary, module map, solve walkthrough"`

### Task 2: ADR scaffolding + six ADRs

**Files:**
- Create: `docs/adr/README.md`, `docs/adr/template.md`, `docs/adr/0001-harness-agent-trust-split.md`, `docs/adr/0002-leak-free-cv-contract.md`, `docs/adr/0003-checkpoint-gate-on-submissions.md`, `docs/adr/0004-append-only-journal-resume.md`, `docs/adr/0005-telemetry-never-crashes.md`, `docs/adr/0006-gemini-retrying-client-cost-ledger.md`

- [ ] **Step 1: Write `template.md`**:

```markdown
# NNNN — Title

**Status:** Accepted | Superseded by NNNN
**Date:** YYYY-MM-DD

## Context
What forced a decision (constraints, alternatives that were on the table).

## Decision
What we chose, stated in one or two sentences. Then the mechanics.

## Consequences
What this makes easy, what it makes hard, what must never regress.
```

- [ ] **Step 2: Write the six ADRs** (Status: Accepted; Date: 2026-06-10 noting "backfilled — decision made during Weeks 1–5"). Each cites the implementing code and spec section. Required content per ADR:
  - **0001:** Context: LLM writes arbitrary Python that the harness must execute. Alternatives: trust-with-review, docker isolation, no in-process execution. Decision: AST denylist lint before any import (`sandbox.lint_module`; forbidden imports/builtins/path-prefixes, alias tracking) + rlimited subprocess for `run_python` (`RLIMIT_AS/CPU/NPROC/FSIZE`). Consequences: cheap, testable, POSIX-only; lint is a denylist so novel escapes are possible — invariant: nothing loads agent code without the lint.
  - **0002:** Context: V1 AutoML leaked via full-dataset transforms; structural guarantee (TransformerMixin) proved subvertible. Decision: temporal guarantee — `cv.train_cv` calls `fit_feature_transformer` with ONLY the fold's training rows; val fold is transformed, never fitted on (spec §6). Consequences: honest CV at the cost of per-fold refits; hard rule #1.
  - **0003:** Context: Kaggle daily submit caps; agent could burn them on regressions. Decision: typed gate (`CheckpointTrigger` enum), first submit always blocks, later submits block on metric-direction-aware regression; four `HandlerMode`s (interactive / auto-safe / stub / deny-all). Consequences: human stays in the loop exactly where money/caps are at stake.
  - **0004:** Context: LLM loops crash; re-running from scratch wastes spend. Decision: append-only JSONL journal, fsync per record, `--resume` rebuilds the Gemini conversation by replaying tool records (`rebuild_conversation`); corrupt trailing lines skipped everywhere. Consequences: any run is resumable; journal is the audit trail; nothing may write to it except `Journal` / `CheckpointHandler`.
  - **0005:** Context: tracking backends (MLflow/OTel/calibration) fail independently of the solve. Decision: every telemetry write is wrapped; failures log to stdlib logging or `<workspace>/mlflow_errors.log` and the loop continues (hard rule #6). Consequences: silent-ish telemetry loss is accepted; solves never die for a dashboard.
  - **0006:** Context: needed a cheap, tool-calling LLM with a multi-turn function protocol. Decision: Gemini (`google-genai`, Content/Part protocol) behind an `LLMClient` interface; `RetryingLLMClient` retries `TransientLLMError` with backoff; every call priced into `cost_ledger.jsonl`. Consequences: model-swappable at the interface; prices table needs manual updates; `--cost-budget` is enforceable.

- [ ] **Step 3: Write `docs/adr/README.md`** — table of the six (number, title, status) + "add one by copying template.md to the next number and listing it here".

- [ ] **Step 4: Commit** — `git add docs/adr && git commit -m "docs: backfill six ADRs + template and index"`

### Task 3: `.claude/` scaffolding

**Files:**
- Create: `.claude/commands/gates.md`, `.claude/commands/solve.md`, `.claude/commands/new-adr.md`, `.claude/commands/harness-review.md`, `.claude/agents/harness-reviewer.md`, `.claude/settings.json`

- [ ] **Step 1: `commands/gates.md`**:

```markdown
---
description: Run the merge gate — full non-slow test tier, ruff, and mypy (harness + agent)
---

Run the three merge gates and report pass/fail for each with the failing output if any:

1. `python -m pytest -m "not slow" -q`
2. `ruff check kaggle_slayer tests`
3. `mypy kaggle_slayer/harness kaggle_slayer/agent`

All three must pass before any commit is considered done. If one fails, stop and fix it before re-running; do not commit on red.
```

- [ ] **Step 2: `commands/solve.md`**:

```markdown
---
description: Run a competition end-to-end (kaggle-slayer solve with the right flags)
---

Run a KaggleSlayer solve. Arguments (workspace path, target column, metric): $ARGUMENTS

1. Confirm `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set; if not, stop and tell the user.
2. Invoke `kaggle-slayer <workspace> --target <col> [--metric <m>]`. Data auto-downloads into `raw/` when missing (slug defaults to the workspace dir name; override with `--competition <slug>`, or skip with `--no-download`).
3. Useful flags: `--resume` (continue an aborted run), `--cost-budget <USD>`, `--max-iterations N`, `--model gemini-2.5-pro`.
4. Report: final status, iterations, spend, and whether `submissions/` got a CSV. Exit codes: 0 done, 1 not finished, 2 setup, 3 resume failure, 4 crash.
```

- [ ] **Step 3: `commands/new-adr.md`**:

```markdown
---
description: Scaffold the next-numbered ADR from the template and add it to the index
---

Create a new ADR for: $ARGUMENTS

1. Find the highest NNNN in `docs/adr/` and use NNNN+1 (zero-padded to 4).
2. Copy `docs/adr/template.md` to `docs/adr/NNNN-<kebab-case-title>.md`; fill Status (Accepted), today's date, and the Context/Decision/Consequences for the topic above.
3. Add a row to the table in `docs/adr/README.md`.
4. Commit both files: `docs: ADR NNNN — <title>`.
```

- [ ] **Step 4: `commands/harness-review.md`**:

```markdown
---
description: Audit the current diff against the six hard rules using the harness-reviewer agent
---

Dispatch the `harness-reviewer` agent on the current changes:

1. Collect the diff: staged + unstaged (`git diff HEAD`), or `$ARGUMENTS` if a ref/range was given.
2. Launch the `harness-reviewer` subagent with that diff and wait for its report.
3. Relay the verdict: each hard rule checked, any violation with file:line evidence, and an overall pass/fail. Block the merge on any violation.
```

- [ ] **Step 5: `agents/harness-reviewer.md`**:

```markdown
---
name: harness-reviewer
description: Audits a diff against KaggleSlayer's six hard rules (leak-free CV, sandbox lint, registries, Kaggle wrapper, journal discipline, telemetry never crashes). Use after changes to kaggle_slayer/ before merging.
tools: Read, Grep, Glob, Bash
---

You audit KaggleSlayer changes against the six hard rules in CLAUDE.md. You receive a diff (or a ref range); read the surrounding code as needed — judge the post-change state, not the diff text alone.

For each rule, actively hunt for violations and cite file:line evidence:

1. **Leak-free CV** — no path may hand full-dataset/val/test rows to agent-written code during fit. Check every call into `fit_feature_transformer` and any new code path in `harness/cv.py` or `handlers/ml.py`.
2. **Lint before import** — any file loaded as agent code must pass `sandbox.lint_module()` first (the sanctioned loader is `cv.load_agent_module`). Flag any new `importlib`/`exec` on workspace files that skips it.
3. **Registries, not inline** — new metrics belong in `harness/registry/metrics.py`, CV strategies in `registry/cv_strategies.py`, checkpoint triggers in `checkpoints.CheckpointTrigger`. Flag inline definitions at call sites.
4. **Kaggle only through the wrapper** — `kaggle.api.*` / `import kaggle` outside `harness/kaggle_client.py` is forbidden; submissions flow only through the gated `submit_kaggle` handler.
5. **Journal discipline** — tool records only via `Journal.log_tool_call`/`log_tool_error`; checkpoint records only via `CheckpointHandler`; submit history in `submissions/leaderboard.jsonl`, never `run_log.jsonl`.
6. **Telemetry never crashes** — every new MLflow/OTel/calibration/crash-report write must be wrapped so failure logs and continues.

Report format: one section per rule — `✅ no violation` or `❌ violation` with evidence and the minimal fix. End with an overall verdict line: `HARNESS-REVIEW: PASS` or `HARNESS-REVIEW: FAIL`.
```

- [ ] **Step 6: `.claude/settings.json`** (checked in; safe read-only/test grants only):

```json
{
  "permissions": {
    "allow": [
      "Bash(pytest:*)",
      "Bash(python -m pytest:*)",
      "Bash(ruff check:*)",
      "Bash(mypy:*)",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(git show:*)",
      "Bash(git ls-files:*)"
    ]
  }
}
```

- [ ] **Step 7: Smoke-check** — `python -c "import json; json.load(open('.claude/settings.json'))"` (valid JSON), and confirm each command file has frontmatter with a `description:`.

- [ ] **Step 8: Commit** — `git add .claude && git commit -m "feat(.claude): gates/solve/new-adr/harness-review commands, harness-reviewer agent, safe permissions"`

### Task 4: CLAUDE.md lean-router rewrite

**Files:**
- Modify: `CLAUDE.md` (full rewrite)

- [ ] **Step 1: Replace CLAUDE.md** with the lean router (~120 lines). Required structure — keep the six hard rules with their original numbering and intent; fix rule 4's stale first sentence; everything moved out gets a pointer:

```markdown
# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

KaggleSlayer V2 — an LLM-agent harness for tabular Kaggle competitions. A Gemini-driven
Solver writes `agent/fe.py` + `agent/model.py`; the trusted Python harness runs leak-free
cross-validation, journals every tool call, gates Kaggle submissions, and ships a CSV.

- **Structure & data flow:** `docs/architecture.md` (module map, solve walkthrough, telemetry, test tiers)
- **Why-decisions:** `docs/adr/` (trust split, leak-free CV, checkpoint gate, journal, telemetry, LLM client)
- **Status, scope, roadmap:** `GOALS.md` — read it before deciding what is in or out of scope
- **History:** specs in `docs/superpowers/specs/`, week-by-week plans in `docs/superpowers/plans/`

## Hard rules

1. **Leak-free CV is the inviolable contract.** `train_cv` only ever passes one fold's
   training data to the agent's `fit_feature_transformer`. Never add a code path that hands
   the full dataset (or any val/test data) to agent-written code during fit. See
   `kaggle_slayer/harness/cv.py`, ADR 0002, and spec §6.
2. **Agent code is never trusted at import time.** Anything loaded as agent-written code
   MUST pass `sandbox.lint_module()` first — load it via `cv.load_agent_module`. See ADR 0001.
3. **The harness owns metrics, CV strategies, and checkpoint triggers.** Extend
   `kaggle_slayer/harness/registry/` or `checkpoints.CheckpointTrigger` — never inline at
   the call site.
4. **Kaggle is touched only through `KaggleClient`** (`harness/kaggle_client.py`), and
   submissions flow only through the checkpoint-gated `submit_kaggle` tool. Direct
   `kaggle.api.*` calls anywhere else are forbidden. See ADR 0003.
5. **The agent never journals directly.** Tool records via `Journal.log_tool_call` /
   `log_tool_error`; checkpoint records via `CheckpointHandler.request`; submit history in
   `submissions/leaderboard.jsonl`, not `run_log.jsonl`. See ADR 0004.
6. **Telemetry never crashes the agent.** MLflow/OTel/calibration failures are caught,
   logged (stdlib `logging`; MLflow → `<workspace>/mlflow_errors.log`), and execution
   continues. See ADR 0005.

## Common commands

\```bash
pip install -e ".[dev,dashboard]"
pytest -m "not slow"                                  # ~5s, 408 tests, no keys — the merge gate
pytest -m slow                                        # opt-in real-Gemini ($)
pytest -m chaos                                       # chaos tier (injected LLM failures)
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness                            # what CI runs
mypy kaggle_slayer/harness kaggle_slayer/agent        # local standard — keep both clean
kaggle-slayer competitions/<name> --target <col> --metric <m>   # full solve (--target required)
kaggle-slayer-dashboard                               # read-only Streamlit dashboard
\```

Or just run `/gates` for the full merge gate. CLI flags worth knowing: `--resume`,
`--cost-budget <USD>`, `--auto-approve {off,safe,all}`, `--model`, `--no-download`,
`--competition <slug>`.

## Conventions

- TDD: write the failing test first. Four test tiers (unit/integration/chaos/slow) — keep them intact.
- Mypy strict and clean on `harness/` AND `agent/` locally; **CI enforces harness only**.
- Errors at boundaries are typed: `CVError`, `ToolError`, `ResumeError`, `TransientLLMError`, `DownloadError`.
- CLI output uses `rich`; library code uses stdlib `logging`; no `print` in library code.
- Tool handlers take `ctx` first (structural contract — see `SolverContext`).
- **Secret hygiene:** `.env` is gitignored and must never be committed (`git add -f` included);
  commit `.env.example`. If a key leaks, rotate it provider-side — scrubbing files does not revoke keys.
- Don't pre-build roadmap items from `GOALS.md`.

## Workflows

- `/gates` — run the merge gate (tests + ruff + mypy)
- `/solve <workspace> <target> [metric]` — run a competition end-to-end
- `/harness-review` — audit the current diff against the hard rules
- `/new-adr <topic>` — scaffold the next ADR

## Files that pin the architecture

Read `docs/architecture.md` §"Files that pin the architecture" before touching
`harness/cv.py`, `harness/checkpoints.py`, `agent/solver.py`, `agent/llm_client.py`,
or `harness/telemetry/`.
\```
```
(Remove the escaping backslashes on the inner code fences when writing the real file.)

- [ ] **Step 2: Verify** — `wc -l CLAUDE.md` ≤ 130; every pointer target exists (`ls docs/architecture.md docs/adr/README.md GOALS.md .claude/commands/gates.md`).

- [ ] **Step 3: Commit** — `git add CLAUDE.md && git commit -m "docs: CLAUDE.md becomes a lean router (content moved to architecture.md + ADRs)"`

### Task 5: README de-drift + docs section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Apply the drift fixes** (each was a confirmed review finding):
  1. Replace every `375` test-count claim with the live number (re-run `python -m pytest -m "not slow" 2>&1 | tail -1` first; 408 as of plan date) — lines ~24, ~45, ~71, ~154.
  2. Fix the integration-tier paragraph (~line 154): CI runs `pytest -m "not slow"` only (which *includes* the integration tests); there is no separate `-m integration` CI job, and markers are applied per-test, not by directory.
  3. Add `data.py` to the repo-layout tree (~line 130): `│   ├─ data.py  # auto-download competition data into raw/`.
  4. Replace the "Week 6" framing for lb_score backfill (~line 40) with a pointer to the GOALS.md roadmap.
  5. Note `--target` is required in the quickstart command table; add `--max-iterations` if absent.
  6. Confirm the OTel claim (~line 116) reads true now that the deps are gone — adjust wording to "no opentelemetry dependency at all".
- [ ] **Step 2: Add a `## Documentation` section** after the quickstart linking: `docs/architecture.md` (how it works), `docs/adr/` (why it's built this way), `GOALS.md` (status/roadmap), `CLAUDE.md` (working in this repo with Claude Code).
- [ ] **Step 3: Verify** — `grep -c 375 README.md` returns 0; `grep -n "data.py" README.md` hits the layout tree.
- [ ] **Step 4: Commit** — `git add README.md && git commit -m "docs(readme): de-drift (test counts, CI claims, data.py, roadmap framing) + docs links"`

### Task 6: GOALS.md update

**Files:**
- Modify: `GOALS.md`

- [ ] **Step 1: Edit:**
  1. "375 tests pass" → live count (line ~15).
  2. Remove "ADRs" and "`.claude/` scaffolding" from the *Out of scope for v1* list and from the *Post-v1 roadmap* table; add a line under "Where we are today": ADRs (`docs/adr/`), `docs/architecture.md`, and `.claude/` commands/agent shipped 2026-06-10 (pulled forward from the roadmap at user request).
  3. Leave the credential-free demo as the headline v1 goal — unchanged.
- [ ] **Step 2: Commit** — `git add GOALS.md && git commit -m "docs(goals): ADRs + .claude scaffolding shipped (pulled forward); refresh counts"`

### Task 7: Honesty pass + gates

- [ ] **Step 1: Re-verify every checkable claim** in README.md / CLAUDE.md / GOALS.md / docs/architecture.md: run `python -m pytest -m "not slow" 2>&1 | tail -1`, `ls` every referenced path, `grep -rn "375" README.md CLAUDE.md GOALS.md docs/` (expect 0 outside historical specs/plans), and diff CLAUDE.md's commands block against `pyproject.toml` entry points + `cli.py` argparse.
- [ ] **Step 2: Run the gates** — `python -m pytest -m "not slow" -q`, `ruff check kaggle_slayer tests`, `mypy kaggle_slayer/harness kaggle_slayer/agent`. All green (docs changes can't break them; the `.claude/` JSON was smoke-checked in Task 3).
- [ ] **Step 3: Fix anything found, amend the relevant commit or add `docs: honesty-pass fixes`.**
