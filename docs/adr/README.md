# Architecture Decision Records

Why KaggleSlayer is built the way it is. Each record captures one decision:
its context, the choice, and what must never regress because of it. The *what*
lives in [`docs/architecture.md`](../architecture.md).

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-harness-agent-trust-split.md) | Harness/agent trust split: lint + sandbox for all agent code | Accepted |
| [0002](0002-leak-free-cv-contract.md) | Leak-free CV: per-fold fit as a temporal guarantee | Accepted |
| [0003](0003-checkpoint-gate-on-submissions.md) | Checkpoint gate on Kaggle submissions | Accepted |
| [0004](0004-append-only-journal-resume.md) | Append-only JSONL journal + resume-by-replay | Accepted |
| [0005](0005-telemetry-never-crashes.md) | Telemetry never crashes the agent | Accepted |
| [0006](0006-gemini-retrying-client-cost-ledger.md) | Gemini behind LLMClient + retrying adapter + cost ledger | Accepted |

## Adding one

Copy [`template.md`](template.md) to `NNNN-<kebab-case-title>.md` (next number,
zero-padded), fill it in, and add a row above — or run `/new-adr <topic>` in a
Claude Code session.
