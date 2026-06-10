---
description: Audit the current diff against the six hard rules using the harness-reviewer agent
---

Dispatch the `harness-reviewer` agent on the current changes:

1. Collect the diff: staged + unstaged (`git diff HEAD`), or use `$ARGUMENTS` as a ref/range if one was given.
2. Launch the `harness-reviewer` subagent with that diff and wait for its report.
3. Relay the verdict: each hard rule checked, any violation with file:line evidence, and the overall `HARNESS-REVIEW: PASS|FAIL` line. A FAIL blocks the merge until fixed.
