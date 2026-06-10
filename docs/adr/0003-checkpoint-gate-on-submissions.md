# 0003 — Checkpoint gate on Kaggle submissions (and other costly actions)

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — decision made during Week 4)

## Context

Kaggle enforces daily submission caps, and a looping agent could burn them on
regressed models; metric changes and budget overruns are similarly consequential.
Fully autonomous submission is cheap to build but expensive to regret; fully
manual submission kills the demo value.

## Decision

A typed gate in `kaggle_slayer/harness/checkpoints.py`. Every consequential action
raises a `CheckpointRequest` with a `CheckpointTrigger` enum value (first submit,
regression submit, non-regression submit, metric change, wall-clock budget, cost
budget, agent-initiated, …) — never an ad-hoc prompt at the call site. Decisions
(`APPROVE`/`DENY`/`ABORT`/`SKIP_CHECK`) are journalled as `kind='checkpoint'`.

Policy: the **first** Kaggle submit always blocks for a human; later submits block
when CV regressed versus the previous best (direction-aware per metric — a worse
rmse is a regression even though the number went up). Four handler modes:
`INTERACTIVE` (default), `AUTO_SAFE` (auto-approves non-regression submits only),
`STUB` (fixed decision, tests/CLI `--auto-approve all` behind
`--i-know-what-im-doing`), `CALLABLE` (custom policy hook).

## Consequences

- The human stays in the loop exactly where money or caps are at stake; everything
  else stays autonomous.
- New consequential actions must extend the trigger enum (hard rule #3), keeping
  the full gate surface enumerable in one file.
- `submit_kaggle` is the only path to Kaggle submission (hard rule #4), so the
  gate cannot be bypassed without touching the wrapper.
