# 0006 — Gemini behind an LLMClient interface, retrying adapter, USD cost ledger

**Status:** Accepted
**Date:** 2026-06-10 (backfilled — client Week 2, retry/ledger Week 5)

## Context

The Solver needs a tool-calling LLM with a multi-turn function protocol, cheap
enough to iterate on (~25 tool calls per solve). Gemini 2.5 Flash validated at
~$0.005–0.02 per synthetic-comp solve. Transient API failures (429/503) are routine
and shouldn't kill runs; uncontrolled spend is the other failure mode.

## Decision

- `agent/llm_client.py:GeminiClient` speaks Gemini's `Content`/`Part` +
  `function_call`/`function_response` protocol behind the minimal `LLMClient`
  interface (`call(messages, tools)`); tool schemas are stripped of
  Gemini-unsupported JSON-Schema keys before send. Transient failures raise the
  typed `TransientLLMError`.
- `agent/retrying_client.py:RetryingLLMClient` is a decorator over any `LLMClient`
  that retries `TransientLLMError` with exponential backoff — retry policy lives
  outside the protocol client.
- Every call is priced into `~/.kaggle_slayer/cost_ledger.jsonl`
  (`agent/cost_ledger.py`, per-model USD price table, per-competition attribution),
  which powers the dashboard cost cards and the `--cost-budget` checkpoint.

## Consequences

- The model is swappable at the `LLMClient` seam (the fake-LLM test client and the
  chaos injector already exploit this).
- The price table is maintained by hand and goes stale when models reprice.
- `--cost-budget` is enforceable because spend is observable per competition;
  approval doubles the budget rather than re-prompting every turn.
- Default model is `gemini-2.5-flash`; Pro is opt-in (`--model gemini-2.5-pro`).
