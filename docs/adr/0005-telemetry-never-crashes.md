# 0005 — Telemetry never crashes the agent

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — decision made during Week 5)

## Context

Week 5 added four tracking surfaces (OTel-style spans, MLflow, the CV↔LB
calibration log, crash reports) plus the cost ledger. Each can fail independently
of the solve — disk full, unwritable home dir, MLflow backend drift — and a solve
that dies because a *dashboard input* failed would waste real Gemini spend on
exactly the runs we most want recorded.

## Decision

Every telemetry write is wrapped: failures are caught, logged via stdlib `logging`
(MLflow failures additionally append to `<workspace>/mlflow_errors.log`), and the
Solver loop continues. The OTel tracer is a ~100-line in-tree JSONL writer
(`telemetry/otel.py`) rather than the `opentelemetry-sdk` — we only need
spans-to-JSONL, and the deps were dropped from `pyproject.toml` entirely; swapping
to OTLP later is one adapter.

## Consequences

- A solve never aborts because a tracking surface is down (hard rule #6).
- The price is silent-ish telemetry loss: missing MLflow runs must be noticed via
  `mlflow_errors.log`, not a crash.
- New telemetry surfaces must follow the same wrap-log-continue pattern, and the
  harness-reviewer agent checks for it on every diff.
