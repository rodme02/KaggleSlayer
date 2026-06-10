---
description: Run a competition end-to-end (kaggle-slayer solve with the right flags)
---

Run a KaggleSlayer solve. Arguments (workspace path, target column, optional metric): $ARGUMENTS

1. Confirm `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set; if not, stop and tell the user (`.env` is loaded automatically).
2. Invoke `kaggle-slayer <workspace> --target <col> [--metric <m>]`. Competition data auto-downloads into `raw/` when missing — the slug defaults to the workspace directory name; override with `--competition <slug>`, or skip the download with `--no-download`.
3. Useful flags: `--resume` (continue an aborted run from run_log.jsonl), `--cost-budget <USD>`, `--max-iterations N`, `--model gemini-2.5-pro` (10x cost), `--auto-approve safe`.
4. Report back: final status, iterations, spend, and whether `submissions/` got a CSV. Exit codes: 0 done · 1 not finished · 2 setup error (keys/download) · 3 resume failure · 4 crash (report under `~/.kaggle_slayer/errors/`).
