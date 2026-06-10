---
description: Run the merge gate — full non-slow test tier, ruff, and mypy (harness + agent)
---

Run the three merge gates and report pass/fail for each, with the failing output if any:

1. `python -m pytest -m "not slow" -q`
2. `ruff check kaggle_slayer tests`
3. `mypy kaggle_slayer/harness kaggle_slayer/agent`

All three must pass before any commit is considered done. If one fails, stop and fix it before re-running; do not commit on red.
