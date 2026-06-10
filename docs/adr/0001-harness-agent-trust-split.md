# 0001 — Harness/agent trust split: lint + sandbox for all agent code

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — decision made during Weeks 1–4)

## Context

The Solver has an LLM write arbitrary Python (`agent/fe.py`, `agent/model.py`,
ad-hoc `run_python` snippets) that the harness must then execute in-process or as a
subprocess. The adversary is not a malicious human but *accidental* agent code:
reading the full dataset inside a transformer (leakage), writing or deleting files
outside the workspace, exhausting memory/CPU, or shelling out.

Alternatives considered: trust-with-human-review (kills autonomy), full container
isolation per call (heavy, slow for a per-fold fit loop), and never executing agent
code in-process (makes the leak-free CV contract impossible to enforce cheaply).

## Decision

Two complementary mechanisms, both in `kaggle_slayer/harness/sandbox.py`:

1. **AST denylist lint before any import.** `lint_module()` parses the file and
   rejects forbidden module attribute chains (`os.remove`, `subprocess.*`, …,
   alias-aware so `import os as o` is still caught), forbidden builtins
   (`eval`/`exec`/`compile`/`__import__`), and forbidden path-literal prefixes
   (e.g. writes into `raw/`). The one sanctioned loader is
   `cv.load_agent_module()`, which lints first and raises `CVError` on violation.
2. **Resource-limited subprocess** for `run_python`: `RLIMIT_AS`, `RLIMIT_CPU`,
   `RLIMIT_NPROC`, `RLIMIT_FSIZE` via `run_subprocess()` (POSIX-only by design).

## Consequences

- Cheap (an AST parse) and unit-testable; no Docker dependency.
- A denylist can be escaped by novel patterns — the protection is *defense in
  depth* with the loader as the choke point. Hard rule #2: nothing imports agent
  code without the lint; any new load path must go through `load_agent_module`.
- POSIX-only resource limits mean Windows is unsupported for `run_python`.
