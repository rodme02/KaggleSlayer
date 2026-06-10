# 0004 — Append-only JSONL journal + resume-by-replay

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — journal Week 1, resume Week 4)

## Context

Agent runs die mid-flight (crashes, quota walls, Ctrl-C) after real Gemini spend.
Restarting from scratch repeats that spend; snapshotting LLM state is brittle and
provider-specific. We also need an audit trail of every tool call for the
dashboard, behavior metrics, and debugging.

## Decision

One append-only JSONL journal per workspace (`run_log.jsonl`), written via
`kaggle_slayer/harness/journal.py:Journal` with flush+fsync per record so a crash
immediately after a call loses nothing. Records are `tool_call` / `tool_error`
(via `log_tool_call`/`log_tool_error` only) and `checkpoint` (via
`CheckpointHandler` only). `--resume` rebuilds the LLM conversation by replaying
the journal (`resume.rebuild_conversation`): each record becomes a
model(function_call) + tool(function_response) message pair, preserving original
tool-call ids when journalled; checkpoint records are skipped; a finished run
(`done`) refuses to resume.

All JSONL readers (journal, notes, calibration, cost ledger, leaderboard) tolerate
a corrupt trailing line — the crash-mid-write case the fsync discipline implies.

## Consequences

- Any aborted run is resumable for the cost of replaying text, not re-running tools.
- The journal doubles as the analytics substrate (behavior metrics, dashboard).
- Hard rule #5: nothing writes journal records except `Journal` and
  `CheckpointHandler`; submit history lives in `submissions/leaderboard.jsonl`.
- Resume fidelity depends on journalled result summaries matching what the LLM saw
  — the solver journals the same truncation it feeds back to the model.
