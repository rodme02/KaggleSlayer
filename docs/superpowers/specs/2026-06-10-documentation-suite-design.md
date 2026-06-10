# Documentation Suite for Claude Code-Driven Development — Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-06-10
**Author:** Rodrigo Medeiros (with Claude)
**Relates to:** `2026-05-14-llm-agent-harness-design.md`, `GOALS.md` (pulls ADRs + `.claude/`
scaffolding forward from the post-v1 roadmap)

---

## 1. Goal

Restructure the project's documentation into a layered, drift-resistant suite optimized
for development driven by Claude Code:

- **CLAUDE.md becomes a lean router** (~120 lines): hard rules, commands, conventions,
  and pointers. It is loaded into every session, so it stays small and stable.
- **Deep content moves to on-demand documents**: `docs/architecture.md` for the "what
  and how", `docs/adr/` for the "why".
- **Repeatable workflows become executable artifacts** under `.claude/` (slash commands,
  a reviewer subagent, a permissions allowlist).
- **Every fact lives in exactly one place.** Status lives in GOALS.md, structure in
  architecture.md, rationale in ADRs, rules in CLAUDE.md. Other documents link instead
  of repeating.

This pulls two items forward from the post-v1 roadmap (ADRs, `.claude/` scaffolding) at
the user's explicit request; GOALS.md is updated so the roadmap stays honest.

## 2. Scope

**In scope:**

- `docs/architecture.md` (new)
- `docs/adr/` — six backfilled ADRs + index + template (new)
- `.claude/commands/` — four slash commands (new)
- `.claude/agents/harness-reviewer.md` (new)
- `.claude/settings.json` — permissions allowlist (new)
- CLAUDE.md rewrite to the lean-router shape (hard rules preserved verbatim)
- README.md drift fixes + docs link tree
- GOALS.md roadmap update

**Out of scope (YAGNI):**

- CONTRIBUTING.md — solo portfolio project; conventions live in CLAUDE.md.
- Generated API reference (Sphinx/mkdocs) — no consumers; the architecture doc covers it.
- Rewriting the historical specs/plans under `docs/superpowers/` — they are immutable
  records of how the project was built.
- The credential-free demo and other GOALS.md roadmap items not named above.

## 3. Components

### 3.1 `docs/architecture.md`

The deep map a fresh Claude session (or human) loads on demand. Sections:

1. **Trust boundary** — the harness (`kaggle_slayer/harness/`) is trusted code; agent
   output (`agent/fe.py`, `agent/model.py`) is never trusted: AST-linted before import,
   executed under resource limits, fed only fold-local data during fit.
2. **Module map** — the full layout tree currently inlined in CLAUDE.md, moved here and
   annotated (one line per module: responsibility + key types).
3. **End-to-end solve walkthrough** — CLI entry → auto-download (`harness/data.py`) →
   context build → Solver reason-act loop → tool dispatch → leak-free `train_cv` →
   checkpoint gate → `submit_kaggle` → journal/telemetry side effects at each step.
4. **On-disk workspace layout** — `competitions/<name>/` contents and the global
   `~/.kaggle_slayer/` surfaces (calibration log, error reports, mlruns).
5. **Telemetry surfaces** — OTel JSONL, MLflow, calibration, crash reports, behavior
   metrics; the never-crash contract.
6. **Test architecture** — the four tiers (unit / integration / chaos / slow), what each
   may touch, and the fixtures (`fake_llm.py`, `synthetic_comp.py`).
7. **Files that pin the architecture** — the read-before-touching list, moved from
   CLAUDE.md.

### 3.2 `docs/adr/` — Architecture Decision Records

Lightweight format per record: **Status / Context / Decision / Consequences** (MADR-style,
one file each, numbered). Backfilled set:

| # | Title | Captures |
| --- | --- | --- |
| 0001 | Harness/agent trust split | Why agent code is sandboxed (AST lint + subprocess rlimits), threat model |
| 0002 | Leak-free CV contract | Per-fold `fit_feature_transformer`, why full-dataset fit is forbidden; carried from V1 |
| 0003 | Checkpoint gate on Kaggle submissions | Why submits are gated, the four modes, score-regression blocking |
| 0004 | Append-only JSONL journal + resume-by-replay | Why JSONL, why replay instead of snapshots |
| 0005 | Telemetry never crashes the agent | Why all tracking failures are swallowed + where they are logged |
| 0006 | Gemini as the LLM + retrying client + cost ledger | Why Gemini 2.5, `TransientLLMError` boundary, per-call USD ledger |

Plus `docs/adr/README.md` (index + how to add one) and `docs/adr/template.md`.
ADRs record decisions already made; they cite the spec sections and code they describe.

### 3.3 `.claude/` scaffolding

- **`commands/gates.md`** — run the merge gate: `pytest -m "not slow"`, `ruff check`,
  `mypy kaggle_slayer/harness kaggle_slayer/agent`; report pass/fail per gate.
- **`commands/solve.md`** — run a competition end-to-end: takes workspace/target/metric,
  knows the common flags (`--competition`, `--no-download`, `--cost-budget`, `--resume`).
- **`commands/new-adr.md`** — scaffold the next-numbered ADR from `template.md` and add
  it to the index.
- **`commands/harness-review.md`** — dispatch the harness-reviewer subagent on the
  current diff.
- **`agents/harness-reviewer.md`** — subagent prompt that audits a diff against the six
  hard rules (leak-free CV, lint-before-import, registry-not-inline, Kaggle-only-through-
  wrapper, journal-only-through-Journal, telemetry-never-crashes) and reports violations
  with file:line evidence.
- **`.claude/settings.json`** — checked-in permissions allowlist limited to safe,
  routine commands: pytest, ruff, mypy, and read-only git (`status`, `diff`, `log`,
  `show`, `ls-files`). No write/network grants.

### 3.4 CLAUDE.md (lean router)

Target shape (~120 lines):

- **What this is** — 3 lines + pointers to `docs/architecture.md` and `GOALS.md`.
- **Hard rules** — the six, preserved verbatim (they are load-bearing and stay in the
  always-loaded file).
- **Common commands** — install, test tiers, lint/typecheck, solve, dashboard.
- **Conventions** — TDD, typed exceptions at boundaries, rich-vs-logging, registry
  extension points, secret hygiene, mypy scope note.
- **Pointers** — architecture.md (structure), adr/ (rationale), GOALS.md (status/scope),
  `.claude/commands/` (workflows), specs/plans (history).

Removed from CLAUDE.md (moved, not deleted): layout tree → architecture.md §2; "files
that pin the architecture" → architecture.md §7; telemetry/MLflow detail → architecture
.md §5; "what's coming" prose → already lives in GOALS.md, keep one pointer line.

### 3.5 README.md and GOALS.md

- README: fix every stale claim found by the review's docs-drift dimension (e.g. test
  counts), keep the portfolio-quality quickstart, add a short "Documentation" section
  linking architecture.md / ADRs / GOALS.md / CLAUDE.md.
- GOALS.md: move ADRs and `.claude/` scaffolding out of the post-v1 roadmap table into
  shipped state; note that docs were restructured (status claims stay accurate).

## 4. Ordering and process

1. The full-project review's confirmed **easy fixes land first** (separate commits) so
   docs describe the post-cleanup codebase.
2. Docs work follows the repo's plan-then-execute convention: this spec → implementation
   plan under `docs/superpowers/plans/` → subagent-driven execution, commits to main.
3. Suggested implementation order inside the plan: architecture.md → ADRs → `.claude/`
   scaffolding → CLAUDE.md slim-down → README → GOALS.md (router last, so its pointers
   target files that already exist).

## 5. Verification

Prose has no unit tests; verification is:

- **Docs-honesty pass** — after everything lands, a reviewer re-checks every checkable
  claim in README / CLAUDE.md / GOALS.md / architecture.md against the code, CI config,
  and a live `pytest --collect-only` count.
- **Gates stay green** — `pytest -m "not slow"`, ruff, mypy unchanged (docs-only changes
  plus `.claude/` files should not affect them).
- **Command smoke check** — `/gates` is invoked once to confirm the command file works.

## 6. Hard-rule compliance

No code paths change. `.claude/settings.json` grants only read-only/test commands, so it
cannot enable a hard-rule violation. CLAUDE.md keeps all six hard rules verbatim in the
always-loaded file.
