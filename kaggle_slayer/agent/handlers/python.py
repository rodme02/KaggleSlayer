"""run_python handler — sandbox-and-execute escape hatch for the agent.

Workflow:
  1. Write the code to workspace.scratch_dir/run_<ts>.py (handled by sandbox.run_subprocess).
  2. Lint the script via sandbox.lint_module — rejected snippets never run.
  3. Invoke sandbox.run_subprocess with RLIMIT_AS + timeout.
  4. Format a string summary the LLM can read.
  5. Prune older run_*.py files so a long run does not accumulate hundreds
     of scratch scripts (F2).

The summary intentionally caps output at ~8 KB so a runaway `print` loop
inside the sandbox cannot balloon the conversation history.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import sandbox

_OUTPUT_CAP_CHARS: int = 8000
_SCRATCH_KEEP_LAST: int = 20


def _truncate(label: str, text: str) -> str:
    if len(text) <= _OUTPUT_CAP_CHARS:
        return f"{label}={text!r}"
    head = text[:_OUTPUT_CAP_CHARS]
    return f"{label}={head!r} [+{len(text) - _OUTPUT_CAP_CHARS} chars truncated]"


def run_python(
    ctx: Any,
    *,
    code: str,
    timeout_s: int = 60,
    memory_mb: int = 2048,
) -> str:
    """Lint, then run `code` in a resource-limited subprocess."""
    # Write the script first so lint sees the actual file path. run_subprocess
    # also writes it, but we lint a freshly-written copy so lint failures don't
    # leave a phantom run_<ts>.py with no execution trace.
    workspace = ctx.workspace
    workspace.scratch_dir.mkdir(parents=True, exist_ok=True)
    import datetime as dt

    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H%M%S_%f")
    script_path = workspace.scratch_dir / f"lint_{stamp}.py"
    script_path.write_text(code)

    lint = sandbox.lint_module(script_path)
    if not lint.ok:
        # Remove the failed-lint phantom so scratch/ stays useful.
        script_path.unlink(missing_ok=True)
        violations = "; ".join(lint.violations[:5])
        raise ToolError(f"lint failed: {violations}")

    # Lint passed — remove the duplicate; run_subprocess writes its own copy.
    script_path.unlink(missing_ok=True)

    result = sandbox.run_subprocess(
        code=code,
        workspace=workspace,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
    )

    # F2: keep scratch/ from growing without bound across a long agent run.
    # Pruning failures must NOT sink the tool call — the tool result already
    # holds the user-visible outcome.
    with contextlib.suppress(OSError):
        _prune_scratch(workspace.scratch_dir, keep_last=_SCRATCH_KEEP_LAST)

    killed = f", killed={result.killed_reason}" if result.killed_reason else ""
    return (
        f"returncode={result.returncode}{killed}, "
        f"{_truncate('stdout', result.stdout)}, "
        f"{_truncate('stderr', result.stderr)}"
    )


def _prune_scratch(scratch_dir: Path, *, keep_last: int = _SCRATCH_KEEP_LAST) -> None:
    """Retain only the `keep_last` most recent run_*.py files in scratch_dir.

    Sorts by mtime (newest first), unlinks the tail. Non-run files are
    untouched. Errors during a single unlink are tolerated so a stuck
    file does not abort the entire prune.
    """
    if not scratch_dir.exists():
        return
    runs = list(scratch_dir.glob("run_*.py"))
    if len(runs) <= keep_last:
        return
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in runs[keep_last:]:
        with contextlib.suppress(OSError):
            stale.unlink()
