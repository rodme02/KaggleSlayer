# 0002 — Leak-free CV: per-fold fit is a temporal guarantee owned by the harness

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — carried from V1's post-mortem, implemented Week 1)

## Context

The archived V1 AutoML pipeline leaked validation information through feature
engineering fitted on the full dataset, producing CV scores that didn't transfer
to the leaderboard. V1's fix attempt — a structural guarantee via sklearn's
`TransformerMixin` — was easy to subvert by accident (any transformer could still
close over the full frame). An LLM agent writing the transformer makes accidental
leakage *more* likely, not less.

## Decision

The harness owns CV end to end (`kaggle_slayer/harness/cv.py:train_cv`) and makes
the guarantee *temporal*: on each fold, the agent's `fit_feature_transformer` is
called with **only that fold's training rows**. The validation fold is transformed
by the already-fitted transformer but never seen at fit time. The agent cannot run
CV any other way — `train_cv` is the only tool that produces a CV score, and the
spec (§6) pins the contract.

## Consequences

- CV scores are honest by construction; the CV↔LB calibration log stays meaningful.
- Cost: the transformer refits once per fold.
- Hard rule #1 (inviolable): no code path may hand the full dataset, or any
  val/test rows, to agent-written code during fit. Any refactor of `cv.py` must
  re-read spec §6 first.
