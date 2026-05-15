# Week 4 — Checkpoint gate, resume, hardened sandbox, submit_kaggle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the spec's §8 (sandbox), §9 (checkpoints), §11.3 (failure-recovery), §12 (resume), and the missing §7 tools (`run_python`, `set_metric`, `submit_kaggle`, `request_human_approval`). At the end of the week, `kaggle-slayer competitions/<comp> --resume` reconstructs an aborted run; submitting to Kaggle goes through a typed human-in-the-loop gate; oversize-memory or runaway code in `run_python` is killed by a real OS-level subprocess sandbox; and the daily cost-budget gate fires when the LLM bill crosses the threshold the user set on the CLI.

**Architecture:** A small `harness/checkpoints.py` module owns the gate UX (typed `Decision` enum, `rich`-styled prompt, journalled outcome). The Solver loop consults it at named trigger points (wall-clock budget, cost-budget, agent-initiated `request_human_approval`) and the gated tool handlers (`submit_kaggle`, `set_metric`) consult it inline. `run_python` lands as a tool whose handler writes the agent's code to `agent/scratch/<ts>.py`, lints it via the existing `sandbox.lint_module`, then runs it through a new `sandbox.run_subprocess(...)` that sets `RLIMIT_AS`/`RLIMIT_CPU` and forces `cwd=workspace.root`. Resume is a pure function over `run_log.jsonl`: walks the records, emits a `list[Message]` that mirrors the original conversation, returns it for the Solver to seed `solve()` with.

**Tech Stack:** Existing harness modules (workspace, journal, cv, sandbox, kaggle_client). `resource.setrlimit` (POSIX) for memory caps. `subprocess.run(..., timeout=...)` for wall-clock cap. `rich.prompt.Prompt` for the gate UI. Gemini SDK is unchanged from Week 3.

**Acceptance:** unit tier green (~25 new tests), integration tier green (fake-LLM checkpoint flow + resume mid-run), slow tier passes a single real-Gemini run that exercises a checkpoint approval and a (mocked) submit_kaggle. Coverage on new harness/agent code ≥ 85%. mypy strict on `harness/` + `agent/` clean.

---

## File map

**Created this week:**
- `kaggle_slayer/harness/checkpoints.py` — `Decision` enum, `CheckpointHandler` (interactive + auto modes), `CheckpointTrigger` enum, gate-prompt logic, journal writes
- `kaggle_slayer/agent/handlers/python.py` — `run_python` handler (lint → subprocess → captured stdout/stderr/returncode)
- `tests/unit/test_sandbox_subprocess.py`
- `tests/unit/test_handlers_python.py`
- `tests/unit/test_checkpoints.py`
- `tests/unit/test_handlers_set_metric.py`
- `tests/unit/test_handlers_submit_kaggle.py`
- `tests/unit/test_resume_rebuild.py`
- `tests/integration/test_checkpoint_flow.py`
- `tests/integration/test_resume_flow.py`
- `tests/integration/test_solver_real_gemini_checkpoint.py` (slow tier, opt-in)

**Modified:**
- `kaggle_slayer/harness/sandbox.py` — add `run_subprocess(code, *, workspace, timeout_s, memory_mb) -> SubprocessResult`
- `kaggle_slayer/harness/resume.py` — add `rebuild_conversation(workspace) -> list[Message]` and `ResumeError`
- `kaggle_slayer/agent/handlers/ml.py` — add `set_metric`, `submit_kaggle`; track `ctx.best_cv_mean` after each `train_cv`
- `kaggle_slayer/agent/handlers/__init__.py` — register `run_python`, `set_metric`, `submit_kaggle`, `request_human_approval` (registry now exposes 13 tools)
- `kaggle_slayer/agent/solver.py` — accept `resume_from: list[Message] | None`; wire wall-clock-budget and cost-budget to the checkpoint gate; add `best_cv_mean` to `SolverContext`
- `kaggle_slayer/harness/journal.py` — bump `result_summary` truncation from 200 chars to 8000 chars so resume has high-fidelity tool results
- `kaggle_slayer/cli.py` — add `--resume`, `--cost-budget`, `--auto-approve` flags; wire to Solver
- `kaggle_slayer/agent/prompts/system.md` — document the four new tools and what `request_human_approval` is for

---

## Task 1: Resource-limited subprocess in `sandbox.py`

Add the OS-level isolation layer the spec §8 calls for: a `run_subprocess` helper that takes Python source, writes it to a temp file under the workspace's scratch dir, runs `python -c <script>` with `cwd=workspace.root`, `RLIMIT_AS` set to a memory cap, and a wall-clock `timeout`. Returns stdout, stderr, returncode, and a "killed" flag distinguishing timeouts from clean exits.

**Files:**
- Modify: `kaggle_slayer/harness/sandbox.py`
- Create: `tests/unit/test_sandbox_subprocess.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_sandbox_subprocess.py`:

```python
"""Tests for kaggle_slayer.harness.sandbox.run_subprocess."""

from __future__ import annotations

import sys

import pytest

from kaggle_slayer.harness import sandbox
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_run_subprocess_returns_stdout(ws):
    result = sandbox.run_subprocess(
        code="print('hello, world')",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert "hello, world" in result.stdout
    assert result.stderr == ""
    assert result.killed_reason is None


def test_run_subprocess_captures_stderr_and_nonzero_exit(ws):
    result = sandbox.run_subprocess(
        code="import sys; sys.stderr.write('boom\\n'); sys.exit(3)",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 3
    assert "boom" in result.stderr


def test_run_subprocess_kills_on_timeout(ws):
    """Infinite loop must be killed by timeout_s; returncode is None or signal."""
    result = sandbox.run_subprocess(
        code="while True:\n    pass",
        workspace=ws,
        timeout_s=1,
        memory_mb=256,
    )
    assert result.killed_reason == "timeout"
    # Should record a returncode != 0 (signal or None depending on platform)
    assert result.returncode != 0


def test_run_subprocess_runs_with_workspace_cwd(ws):
    """The subprocess's cwd must be the workspace root, not the test cwd."""
    result = sandbox.run_subprocess(
        code="import os; print(os.getcwd())",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert str(ws.root.resolve()) in result.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS not enforced reliably on macOS")
def test_run_subprocess_kills_on_memory_overflow(ws):
    """Allocating a huge bytearray triggers the memory limit on Linux."""
    code = (
        "data = bytearray(900 * 1024 * 1024)\n"  # 900 MB
        "print('alloc_done')\n"
    )
    result = sandbox.run_subprocess(
        code=code,
        workspace=ws,
        timeout_s=15,
        memory_mb=128,  # 128 MB cap — much less than 900 MB attempt
    )
    # On Linux RLIMIT_AS forces MemoryError or kill; alloc_done must not print
    assert "alloc_done" not in result.stdout


def test_run_subprocess_writes_script_to_scratch_dir(ws):
    """The temp script lives under workspace.scratch_dir for post-hoc debugging."""
    sandbox.run_subprocess(
        code="x = 1\nprint(x)",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    scripts = list(ws.scratch_dir.glob("run_*.py"))
    assert len(scripts) >= 1
    assert "x = 1" in scripts[0].read_text()
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_sandbox_subprocess.py -v
```

Expected: `AttributeError: module 'kaggle_slayer.harness.sandbox' has no attribute 'run_subprocess'`.

- [ ] **Step 3: Add `run_subprocess` to `kaggle_slayer/harness/sandbox.py`**

Append to the file (after the existing `lint_module` function):

```python
import datetime as _dt
import resource as _resource  # noqa: PLC0415 — POSIX-only; we don't support Windows
import subprocess as _subprocess
import sys as _sys

from kaggle_slayer.harness.workspace import Workspace as _Workspace


@dataclass(frozen=True)
class SubprocessResult:
    """Outcome of run_subprocess: captured streams + classification.

    killed_reason is one of:
      None      — clean exit (success or non-zero rc with normal termination)
      "timeout" — wall-clock cap hit
      "memory"  — RLIMIT_AS triggered the kill (heuristic: SIGKILL on Linux)
    """

    returncode: int
    stdout: str
    stderr: str
    killed_reason: str | None
    script_path: Path


def _preexec_setrlimit(memory_bytes: int, cpu_seconds: int) -> None:
    """preexec_fn payload — POSIX-only. Sets per-process limits before exec."""
    # Memory cap (address space). macOS often ignores RLIMIT_AS; document elsewhere.
    _resource.setrlimit(_resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    # CPU-time cap as a backstop against busy loops the wall-clock timeout might
    # race with (e.g., kernel scheduling slop). Same value as wall-clock for now.
    _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def run_subprocess(
    *,
    code: str,
    workspace: _Workspace,
    timeout_s: int = 60,
    memory_mb: int = 2048,
) -> SubprocessResult:
    """Run Python `code` in an isolated subprocess scoped to the workspace.

    Writes `code` to `workspace.scratch_dir/run_<ts>.py` for debuggability,
    invokes `python <script>` with `cwd=workspace.root`, applies RLIMIT_AS
    and RLIMIT_CPU via preexec_fn, and enforces wall-clock via subprocess
    `timeout`. Returns a SubprocessResult capturing stdout/stderr/returncode
    and a `killed_reason` string for timeout/memory kills.

    The caller is responsible for AST-linting `code` first (e.g., by writing
    it to a file and calling `lint_module`). This function does not lint.
    """
    workspace.scratch_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d_%H%M%S_%f")
    script_path = workspace.scratch_dir / f"run_{stamp}.py"
    script_path.write_text(code)

    memory_bytes = max(1, memory_mb) * 1024 * 1024
    preexec = None
    if _sys.platform != "win32":
        # Bind the two arguments at closure-construction time.
        def preexec() -> None:  # noqa: ANN202 — used only by subprocess
            _preexec_setrlimit(memory_bytes, timeout_s)

    try:
        completed = _subprocess.run(
            [_sys.executable, str(script_path)],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=preexec,
            check=False,
        )
        killed_reason = None
        # Heuristic: SIGKILL (-9) on Linux when RLIMIT_AS fires.
        if completed.returncode == -9:
            killed_reason = "memory"
        return SubprocessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            killed_reason=killed_reason,
            script_path=script_path,
        )
    except _subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return SubprocessResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            killed_reason="timeout",
            script_path=script_path,
        )
```

Also at the top of the file, replace the existing module docstring's last line:

```python
Resource limits (subprocess + setrlimit) are added in Week 4.
```

with:

```python
Resource limits (subprocess + setrlimit) ship in `run_subprocess` below —
used by the `run_python` tool. Note: macOS does not reliably enforce
RLIMIT_AS; the memory cap is a best-effort on Darwin and a hard cap on
Linux.
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_sandbox_subprocess.py -v
```

Expected: 5 passes on macOS (`test_run_subprocess_kills_on_memory_overflow` is skipped); 6 on Linux.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/sandbox.py tests/unit/test_sandbox_subprocess.py
git commit -m "$(cat <<'EOF'
feat(sandbox): add run_subprocess with RLIMIT_AS + RLIMIT_CPU + cwd lock

run_subprocess(code, workspace, timeout_s, memory_mb) writes the source
to workspace/agent/scratch/run_<ts>.py, runs `python <script>` with
cwd=workspace.root and a preexec_fn that calls setrlimit(RLIMIT_AS) and
setrlimit(RLIMIT_CPU). The wall-clock cap is enforced by subprocess's
own timeout. Returns SubprocessResult(returncode, stdout, stderr,
killed_reason, script_path); killed_reason classifies "timeout" /
"memory" so the run_python tool can surface a typed error.

This is the OS-level layer spec §8 calls for. AST lint stays in
lint_module; the run_python handler (Task 2) chains both. macOS notes:
RLIMIT_AS is best-effort on Darwin — documented in the module docstring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `run_python` tool

The agent's escape hatch for plotting, peeks, and debug — never for CV. Lints the supplied code, runs it through `run_subprocess`, returns a string summary the LLM can read.

**Files:**
- Create: `kaggle_slayer/agent/handlers/python.py`
- Create: `tests/unit/test_handlers_python.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_handlers_python.py`:

```python
"""Tests for kaggle_slayer.agent.handlers.python.run_python."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaggle_slayer.agent.handlers import python as ph
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_run_python_executes_simple_code_and_returns_summary(ctx):
    result = ph.run_python(ctx, code="print(2 + 2)")
    assert "stdout=" in result and "4" in result
    assert "returncode=0" in result


def test_run_python_includes_stderr(ctx):
    result = ph.run_python(
        ctx,
        code="import sys; sys.stderr.write('warn\\n'); print('ok')",
    )
    assert "warn" in result and "ok" in result


def test_run_python_propagates_non_zero_exit(ctx):
    result = ph.run_python(ctx, code="import sys; sys.exit(7)")
    assert "returncode=7" in result


def test_run_python_rejects_lint_violations(ctx):
    """The code is AST-linted before exec; os.remove must be rejected."""
    with pytest.raises(ToolError, match="lint"):
        ph.run_python(ctx, code="import os; os.remove('train.csv')")


def test_run_python_rejects_subprocess_imports(ctx):
    with pytest.raises(ToolError, match="lint"):
        ph.run_python(
            ctx,
            code="import subprocess; subprocess.run(['ls'])",
        )


def test_run_python_handles_timeout(ctx):
    result = ph.run_python(ctx, code="while True:\n    pass", timeout_s=1)
    assert "killed=timeout" in result


def test_run_python_caps_output_size(ctx):
    """Tool result must not balloon the conversation history."""
    big = "for _ in range(100000):\n    print('x' * 80)\n"
    result = ph.run_python(ctx, code=big, timeout_s=5)
    # The result string we return to the LLM is capped to 8 KB.
    assert len(result) <= 9000  # 8 KB cap + small header/footer overhead


def test_run_python_writes_script_to_scratch(ctx):
    ph.run_python(ctx, code="print('persisted')")
    scripts = list(ctx.workspace.scratch_dir.glob("run_*.py"))
    assert len(scripts) >= 1
    assert "persisted" in scripts[0].read_text()
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_handlers_python.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.handlers.python`.

- [ ] **Step 3: Create `kaggle_slayer/agent/handlers/python.py`**

```python
"""run_python handler — sandbox-and-execute escape hatch for the agent.

Workflow:
  1. Write the code to workspace.scratch_dir/run_<ts>.py (handled by sandbox.run_subprocess).
  2. Lint the script via sandbox.lint_module — rejected snippets never run.
  3. Invoke sandbox.run_subprocess with RLIMIT_AS + timeout.
  4. Format a string summary the LLM can read.

The summary intentionally caps output at ~8 KB so a runaway `print` loop
inside the sandbox cannot balloon the conversation history.
"""

from __future__ import annotations

from typing import Any

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import sandbox

_OUTPUT_CAP_CHARS: int = 8000


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

    killed = f", killed={result.killed_reason}" if result.killed_reason else ""
    return (
        f"returncode={result.returncode}{killed}, "
        f"{_truncate('stdout', result.stdout)}, "
        f"{_truncate('stderr', result.stderr)}"
    )
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_handlers_python.py -v
```

Expected: 8 passes.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/python.py tests/unit/test_handlers_python.py
git commit -m "$(cat <<'EOF'
feat(agent): add run_python tool — sandboxed escape hatch

run_python(code, timeout_s, memory_mb) chains sandbox.lint_module then
sandbox.run_subprocess: lint failures raise ToolError before exec;
clean snippets run under RLIMIT_AS + RLIMIT_CPU + workspace cwd. The
returned string summary is capped at ~8 KB so a runaway print loop in
the sandbox can't blow up the conversation history.

The tool is the spec §7 'for plotting, peeks, debug — NOT for CV'
surface. CV stays in the harness via train_cv; run_python cannot
shortcut leak-free.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Checkpoint gate module

The `harness/checkpoints.py` module defines the typed gate primitives. The CLI provides a `rich`-styled prompt; tests use a stub. Decisions are journalled.

**Files:**
- Create: `kaggle_slayer/harness/checkpoints.py`
- Create: `tests/unit/test_checkpoints.py`
- Modify: `kaggle_slayer/harness/journal.py` (add `log_checkpoint` method)

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_checkpoints.py`:

```python
"""Tests for kaggle_slayer.harness.checkpoints."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def journal(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return Journal(ws)


def test_decision_enum_has_four_outcomes():
    assert {cp.Decision.APPROVE, cp.Decision.DENY, cp.Decision.ABORT, cp.Decision.SKIP_CHECK}
    assert cp.Decision.APPROVE.value == "approve"


def test_checkpoint_request_dataclass():
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE,
        action="submit submission 'lr_v1' to kaggle competition titanic",
        evidence={"csv_rows": 418, "cv_mean": 0.81},
    )
    assert req.trigger == cp.CheckpointTrigger.SUBMIT_KAGGLE
    assert req.evidence["cv_rows" if False else "csv_rows"] == 418


def test_handler_auto_mode_approves_safe_triggers(journal):
    """Auto-mode 'safe' approves auto-approve cases (per spec §9)."""
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=journal)
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE_NON_REGRESSION,
        action="resubmit (CV improved)",
        evidence={"cv_mean": 0.85, "prev_best": 0.83},
    )
    assert handler.request(req) == cp.Decision.APPROVE


def test_handler_auto_mode_denies_unsafe_triggers(journal):
    """Auto-mode 'safe' denies the always-block triggers (per spec §9)."""
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=journal)
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE_FIRST,
        action="first submission",
        evidence={},
    )
    assert handler.request(req) == cp.Decision.DENY


def test_handler_stub_mode_uses_injected_decision(journal):
    """Stub mode is used by tests: a fixed Decision."""
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SET_METRIC,
        action="change metric to auc",
        evidence={"current": "accuracy", "proposed": "auc"},
    )
    assert handler.request(req) == cp.Decision.APPROVE


def test_handler_journals_every_decision(journal):
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )
    handler.request(cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.WALL_CLOCK_BUDGET,
        action="extend wall-clock budget",
        evidence={"elapsed_s": 5400, "budget_s": 5400},
    ))
    lines = journal.workspace.run_log_path.read_text().splitlines()
    records = [json.loads(l) for l in lines]
    cp_records = [r for r in records if r.get("kind") == "checkpoint"]
    assert len(cp_records) == 1
    assert cp_records[0]["decision"] == "deny"
    assert cp_records[0]["trigger"] == "wall_clock_budget"
    assert cp_records[0]["evidence"]["elapsed_s"] == 5400


def test_callable_mode_invokes_provided_function(journal):
    """The CLI-side prompt uses a function injected via HandlerMode.CALLABLE."""
    seen: list[cp.CheckpointRequest] = []

    def prompt(req: cp.CheckpointRequest) -> cp.Decision:
        seen.append(req)
        return cp.Decision.ABORT

    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt
    )
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.COST_BUDGET,
        action="raise cost budget",
        evidence={"spent_usd": 5.20, "budget_usd": 5.0},
    )
    assert handler.request(req) == cp.Decision.ABORT
    assert len(seen) == 1
    assert seen[0].trigger == cp.CheckpointTrigger.COST_BUDGET
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_checkpoints.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.checkpoints`.

- [ ] **Step 3: Create `kaggle_slayer/harness/checkpoints.py`**

```python
"""Checkpoint gate — typed pause-points where the harness blocks the agent.

Spec §9 defines six triggers (submit_kaggle first / submit_kaggle regression /
set_metric / wall-clock budget / cost budget / agent-initiated). Each emits a
CheckpointRequest; the CheckpointHandler turns that into a Decision.

The handler has four modes:
  INTERACTIVE   — rich CLI prompt (used by the real CLI; not unit-tested)
  AUTO_SAFE     — auto-approves the spec's 'auto-approve' triggers, denies the rest
  STUB          — returns a fixed Decision (used by tests)
  CALLABLE      — calls a user-supplied prompt_fn (used by the CLI; testable)

Every Decision is journalled to run_log.jsonl as kind='checkpoint'.
"""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal


class CheckpointTrigger(enum.Enum):
    """The named gate points from spec §9 + agent-initiated."""

    SUBMIT_KAGGLE_FIRST = "submit_kaggle_first"
    SUBMIT_KAGGLE_REGRESSION = "submit_kaggle_regression"
    SUBMIT_KAGGLE_NON_REGRESSION = "submit_kaggle_non_regression"
    SET_METRIC = "set_metric"
    WALL_CLOCK_BUDGET = "wall_clock_budget"
    COST_BUDGET = "cost_budget"
    MEMORY_SUSTAINED = "memory_sustained"
    AGENT_INITIATED = "agent_initiated"


class Decision(enum.Enum):
    APPROVE = "approve"
    DENY = "deny"
    ABORT = "abort"
    SKIP_CHECK = "skip_check"  # approve + don't ask again this run


class HandlerMode(enum.Enum):
    INTERACTIVE = "interactive"
    AUTO_SAFE = "auto_safe"
    STUB = "stub"
    CALLABLE = "callable"


# Triggers that auto_safe approves without prompting.
_AUTO_SAFE_APPROVES: frozenset[CheckpointTrigger] = frozenset({
    CheckpointTrigger.SUBMIT_KAGGLE_NON_REGRESSION,
})


@dataclass(frozen=True)
class CheckpointRequest:
    trigger: CheckpointTrigger
    action: str
    evidence: dict[str, Any] = field(default_factory=dict)


class CheckpointHandler:
    """Bound to one Journal; dispatches per-trigger to the configured mode."""

    def __init__(
        self,
        *,
        mode: HandlerMode,
        journal: Journal,
        stub_decision: Decision | None = None,
        prompt_fn: Callable[[CheckpointRequest], Decision] | None = None,
    ) -> None:
        self.mode = mode
        self.journal = journal
        self.stub_decision = stub_decision
        self.prompt_fn = prompt_fn
        self._skipped: set[CheckpointTrigger] = set()
        if mode == HandlerMode.STUB and stub_decision is None:
            raise ValueError("STUB mode requires stub_decision")
        if mode == HandlerMode.CALLABLE and prompt_fn is None:
            raise ValueError("CALLABLE mode requires prompt_fn")

    def request(self, req: CheckpointRequest) -> Decision:
        decision = self._decide(req)
        if decision == Decision.SKIP_CHECK:
            self._skipped.add(req.trigger)
        self._journal(req, decision)
        return decision

    def _decide(self, req: CheckpointRequest) -> Decision:
        if req.trigger in self._skipped:
            return Decision.APPROVE
        if self.mode == HandlerMode.STUB:
            assert self.stub_decision is not None  # checked in __init__
            return self.stub_decision
        if self.mode == HandlerMode.AUTO_SAFE:
            return Decision.APPROVE if req.trigger in _AUTO_SAFE_APPROVES else Decision.DENY
        if self.mode == HandlerMode.CALLABLE:
            assert self.prompt_fn is not None
            return self.prompt_fn(req)
        if self.mode == HandlerMode.INTERACTIVE:
            return _interactive_prompt(req)
        raise RuntimeError(f"unhandled mode: {self.mode}")

    def _journal(self, req: CheckpointRequest, decision: Decision) -> None:
        self.journal._append(  # noqa: SLF001 — checkpoint kind is part of journal contract
            self.journal.workspace.run_log_path,
            {
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "kind": "checkpoint",
                "trigger": req.trigger.value,
                "action": req.action,
                "evidence": req.evidence,
                "decision": decision.value,
            },
        )


def _interactive_prompt(req: CheckpointRequest) -> Decision:
    """Default rich-styled prompt — never reached in tests."""
    from rich.console import Console  # noqa: PLC0415
    from rich.panel import Panel  # noqa: PLC0415
    from rich.prompt import Prompt  # noqa: PLC0415

    console = Console()
    body = f"[bold]{req.action}[/]\n\n"
    for k, v in req.evidence.items():
        body += f"  • {k}: {v}\n"
    console.print(Panel(body, title=f"Checkpoint: {req.trigger.value}", border_style="yellow"))
    choice = Prompt.ask(
        "Decision",
        choices=["y", "n", "a", "s"],
        default="n",
        show_choices=True,
    )
    return {
        "y": Decision.APPROVE,
        "n": Decision.DENY,
        "a": Decision.ABORT,
        "s": Decision.SKIP_CHECK,
    }[choice]
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_checkpoints.py -v
```

Expected: 7 passes.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/checkpoints.py tests/unit/test_checkpoints.py
git commit -m "$(cat <<'EOF'
feat(harness): add checkpoints module — typed gate + journalled decisions

CheckpointTrigger enum codifies spec §9's six trigger points plus an
agent-initiated entry. Decision enum: approve / deny / abort / skip_check.
HandlerMode supports four routings: interactive (rich CLI), auto_safe
(spec's defaults — approves SUBMIT_KAGGLE_NON_REGRESSION, denies the
rest), stub (tests), callable (user-supplied prompt_fn, used by the CLI
adapter).

Every decision lands in run_log.jsonl as kind='checkpoint' with trigger,
action, evidence, and outcome. skip_check de-suppresses future requests
for the same trigger in the current run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `set_metric` tool (always gated)

Per spec §7: `set_metric` requires checkpoint approval. The agent uses this when it disagrees with the metric Kaggle parsed (e.g., Kaggle says "accuracy" but the comp is actually weighted F1).

**Files:**
- Modify: `kaggle_slayer/agent/handlers/ml.py` — add `set_metric` function
- Modify: `kaggle_slayer/agent/solver.py` — add `checkpoint_handler` to `SolverContext`
- Create: `tests/unit/test_handlers_set_metric.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_handlers_set_metric.py`:

```python
"""Tests for ml_h.set_metric — always checkpoint-gated."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""
    checkpoint_handler: cp.CheckpointHandler | None = None
    best_cv_mean: float | None = None


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    return _Ctx(
        workspace=ws,
        journal=journal,
        checkpoint_handler=cp.CheckpointHandler(
            mode=cp.HandlerMode.STUB,
            journal=journal,
            stub_decision=cp.Decision.APPROVE,
        ),
    )


def test_set_metric_changes_metric_on_approval(ctx):
    result = ml_h.set_metric(ctx, name="auc")
    assert ctx.metric_name == "auc"
    assert "auc" in result


def test_set_metric_rejected_on_deny(ctx, tmp_path):
    """If the checkpoint handler denies, metric must not change."""
    ws = Workspace.create(root=tmp_path / "comp2")
    journal = Journal(ws)
    deny_handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )
    ctx_deny = type(ctx)(workspace=ws, journal=journal, checkpoint_handler=deny_handler)
    original = ctx_deny.metric_name
    with pytest.raises(ToolError, match="denied"):
        ml_h.set_metric(ctx_deny, name="auc")
    assert ctx_deny.metric_name == original


def test_set_metric_validates_known_metric(ctx):
    """Unknown metric is rejected before the checkpoint runs."""
    with pytest.raises(ToolError, match="unknown metric"):
        ml_h.set_metric(ctx, name="bogus_metric")


def test_set_metric_journals_checkpoint_decision(ctx):
    ml_h.set_metric(ctx, name="auc")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(l) for l in lines if json.loads(l).get("kind") == "checkpoint"]
    assert len(cp_records) == 1
    assert cp_records[0]["trigger"] == "set_metric"
    assert cp_records[0]["decision"] == "approve"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_handlers_set_metric.py -v
```

Expected: `AttributeError: module 'kaggle_slayer.agent.handlers.ml' has no attribute 'set_metric'`.

- [ ] **Step 3: Add `set_metric` to `kaggle_slayer/agent/handlers/ml.py`**

Append at the end of the file:

```python
def set_metric(ctx: Any, *, name: str) -> str:
    """Change the scoring metric. Always checkpoint-gated (spec §9)."""
    # Validate before pestering the user.
    try:
        metrics.get(name)
    except KeyError as e:
        raise ToolError(f"unknown metric {name!r}; choose from the registry") from e

    handler = getattr(ctx, "checkpoint_handler", None)
    if handler is None:
        raise ToolError("set_metric is gated but no checkpoint handler is configured")

    from kaggle_slayer.harness import checkpoints as cp  # local import avoids cycles

    decision = handler.request(cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SET_METRIC,
        action=f"change metric from {ctx.metric_name!r} to {name!r}",
        evidence={"current": ctx.metric_name, "proposed": name},
    ))
    if decision == cp.Decision.DENY:
        raise ToolError(f"checkpoint denied set_metric to {name!r}")
    if decision == cp.Decision.ABORT:
        raise ToolError("aborted by user at set_metric checkpoint")
    # APPROVE or SKIP_CHECK — capture the prior name before mutation.
    previous = ctx.metric_name
    ctx.metric_name = name
    return f"metric changed to {name!r} (was {previous!r}, approved by checkpoint)"
```

- [ ] **Step 4: Extend `SolverContext` in `kaggle_slayer/agent/solver.py`**

Find the existing `SolverContext` dataclass and add two new fields after `final_summary: str = ""`:

```python
    checkpoint_handler: Any | None = None  # CheckpointHandler; Any to avoid harness import cycle
    best_cv_mean: float | None = None
```

(Also: the line `checkpoint_handler: Any | None = None` needs `from typing import Any` — which the file already imports.)

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_handlers_set_metric.py tests/unit/test_solver.py -v
```

Expected: new tests pass; existing Solver tests still pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/ml.py kaggle_slayer/agent/solver.py tests/unit/test_handlers_set_metric.py
git commit -m "$(cat <<'EOF'
feat(agent): add set_metric tool — always checkpoint-gated

set_metric(name) validates the proposed metric against the registry,
then routes through ctx.checkpoint_handler with trigger=SET_METRIC. On
APPROVE/SKIP_CHECK the metric is changed; DENY/ABORT raise ToolError
that the Solver loop surfaces back to the LLM (so the agent can pick
a different name or give up). Decision is journalled.

SolverContext gains two fields: checkpoint_handler (typed Any to avoid
the harness→agent import cycle) and best_cv_mean (used by Task 5 to
classify submit_kaggle regression vs improvement).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `submit_kaggle` tool with regression-aware checkpoint

Per spec §9: first submit_kaggle always blocks; subsequent submits auto-approve only when CV did NOT regress vs the best previous CV. The handler tracks `ctx.best_cv_mean` (set by `train_cv`).

**Files:**
- Modify: `kaggle_slayer/agent/handlers/ml.py` — add `submit_kaggle`; have `train_cv` update `ctx.best_cv_mean`
- Modify: `kaggle_slayer/agent/solver.py` — add `kaggle_client` to `SolverContext`
- Create: `tests/unit/test_handlers_submit_kaggle.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_handlers_submit_kaggle.py`:

```python
"""Tests for ml_h.submit_kaggle — checkpoint-gated, regression-aware."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""
    checkpoint_handler: cp.CheckpointHandler | None = None
    best_cv_mean: float | None = None
    kaggle_client: object | None = None
    competition: str = "test-comp"


def _make_ctx(tmp_path, *, stub_decision=cp.Decision.APPROVE):
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    # Place a fake submission CSV so submit_kaggle has something to push
    (ws.submissions_dir / "2026-05-15_001_lr.csv").write_text("id,target\n1,0\n2,1\n")
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=stub_decision
    )
    fake_kaggle = MagicMock()
    return _Ctx(
        workspace=ws, journal=journal, checkpoint_handler=handler,
        kaggle_client=fake_kaggle, competition="test-comp",
    )


def test_submit_kaggle_first_submission_gated_as_first(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    result = ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")
    # Approved → kaggle.submit must have been called once
    ctx.kaggle_client.submit.assert_called_once()
    # And the trigger journalled was SUBMIT_KAGGLE_FIRST
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(l) for l in lines if json.loads(l).get("kind") == "checkpoint"]
    assert cp_records[0]["trigger"] == "submit_kaggle_first"
    assert "test-comp" in result


def test_submit_kaggle_denied_does_not_submit(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    with pytest.raises(ToolError, match="denied"):
        ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")
    ctx.kaggle_client.submit.assert_not_called()


def test_submit_kaggle_subsequent_non_regression(tmp_path):
    """Second submission with same-or-better CV is gated as SUBMIT_KAGGLE_NON_REGRESSION."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    # Simulate a successful first submission already journalled
    ctx.journal._append(  # noqa: SLF001
        ctx.workspace.run_log_path,
        {"ts": "2026-05-15", "kind": "tool_call", "tool": "submit_kaggle",
         "args": {"csv_path": "..", "message": "v1"}, "result_summary": "submitted"},
    )
    ctx.best_cv_mean = 0.80
    # Pretend a v2 CSV exists; the test sets best_cv_mean to a higher value (improved)
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    # The new CV that the LLM just ran is implicitly stored as best_cv_mean already.
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(l) for l in lines if json.loads(l).get("kind") == "checkpoint"]
    # Most recent checkpoint must be a non-regression trigger
    assert cp_records[-1]["trigger"] == "submit_kaggle_non_regression"


def test_submit_kaggle_subsequent_regression(tmp_path):
    """Second submission with worse CV than best is gated as SUBMIT_KAGGLE_REGRESSION."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.journal._append(  # noqa: SLF001
        ctx.workspace.run_log_path,
        {"ts": "2026-05-15", "kind": "tool_call", "tool": "submit_kaggle",
         "args": {"csv_path": "..", "message": "v1"}, "result_summary": "submitted",
         "cv_at_submit": 0.85},  # we track the CV at each submission
    )
    ctx.best_cv_mean = 0.80  # current model is WORSE than 0.85 at the previous submit
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(l) for l in lines if json.loads(l).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_regression"


def test_submit_kaggle_rejects_nonexistent_csv(tmp_path):
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        ml_h.submit_kaggle(ctx, csv_path="submissions/missing.csv", message="x")


def test_submit_kaggle_rejects_path_traversal(tmp_path):
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolError, match="outside"):
        ml_h.submit_kaggle(ctx, csv_path="../escape.csv", message="x")
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_handlers_submit_kaggle.py -v
```

Expected: `AttributeError: 'submit_kaggle'`.

- [ ] **Step 3: Add `submit_kaggle` to `kaggle_slayer/agent/handlers/ml.py`**

Append at the end of the file:

```python
def submit_kaggle(ctx: Any, *, csv_path: str, message: str) -> str:
    """Push a submission CSV to Kaggle. Checkpoint-gated by spec §9.

    Trigger classification:
      - First submission in this workspace → SUBMIT_KAGGLE_FIRST (always blocks)
      - Subsequent, CV regressed vs best → SUBMIT_KAGGLE_REGRESSION (blocks)
      - Subsequent, CV did not regress → SUBMIT_KAGGLE_NON_REGRESSION (auto_safe approves)
    """
    workspace = ctx.workspace
    handler = getattr(ctx, "checkpoint_handler", None)
    kaggle = getattr(ctx, "kaggle_client", None)
    if handler is None:
        raise ToolError("submit_kaggle is gated but no checkpoint handler is configured")
    if kaggle is None:
        raise ToolError("submit_kaggle requires a kaggle_client on the context")

    from kaggle_slayer.agent.handlers.files import _resolve_under  # noqa: PLC0415
    from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

    target_csv = _resolve_under(workspace.root, csv_path)
    if not target_csv.exists():
        raise ToolError(f"submission CSV not found: {csv_path}")

    trigger, prev_cv = _classify_submit_trigger(ctx)
    decision = handler.request(cp.CheckpointRequest(
        trigger=trigger,
        action=f"submit '{target_csv.name}' to kaggle competition {ctx.competition}",
        evidence={
            "csv_path": csv_path,
            "message": message,
            "cv_mean": ctx.best_cv_mean,
            "prev_best_cv": prev_cv,
            "kind": trigger.value,
        },
    ))
    if decision in (cp.Decision.DENY, cp.Decision.ABORT):
        raise ToolError(f"checkpoint denied submit_kaggle ({decision.value})")

    kaggle.submit(ctx.competition, csv_path=target_csv, message=message)
    # Re-journal with cv_at_submit so future regression checks can compare.
    ctx.journal._append(  # noqa: SLF001
        ctx.workspace.run_log_path,
        {
            "ts": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(timespec="seconds"),
            "kind": "tool_call",
            "tool": "submit_kaggle",
            "args": {"csv_path": csv_path, "message": message},
            "result_summary": f"submitted to {ctx.competition}",
            "cv_at_submit": ctx.best_cv_mean,
        },
    )
    return f"submitted '{target_csv.name}' to {ctx.competition!r} (msg={message!r})"


def _classify_submit_trigger(ctx: Any) -> tuple[Any, float | None]:
    """Inspect the journal: was this the first submit, regression, or improvement?

    Returns (CheckpointTrigger, prev_best_cv_seen_at_a_prior_submit).
    """
    from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

    prev_cvs: list[float] = []
    for rec in ctx.journal.iter_records():
        if rec.get("tool") == "submit_kaggle" and rec.get("kind") == "tool_call":
            cv = rec.get("cv_at_submit")
            if isinstance(cv, (int, float)):
                prev_cvs.append(float(cv))
    if not prev_cvs:
        return cp.CheckpointTrigger.SUBMIT_KAGGLE_FIRST, None
    prev_best = max(prev_cvs)
    current = ctx.best_cv_mean if ctx.best_cv_mean is not None else float("-inf")
    if current < prev_best:
        return cp.CheckpointTrigger.SUBMIT_KAGGLE_REGRESSION, prev_best
    return cp.CheckpointTrigger.SUBMIT_KAGGLE_NON_REGRESSION, prev_best
```

- [ ] **Step 4: Update `train_cv` to record `ctx.best_cv_mean`**

Find the existing `train_cv` function in `kaggle_slayer/agent/handlers/ml.py`. Just before the `return summary` line, insert:

```python
    # Track best CV for the regression-aware submit_kaggle gate (spec §9).
    prior = getattr(ctx, "best_cv_mean", None)
    if prior is None or result.mean > prior:
        ctx.best_cv_mean = float(result.mean)
```

- [ ] **Step 5: Add `kaggle_client` and `competition` to `SolverContext`**

In `kaggle_slayer/agent/solver.py`, find the `SolverContext` dataclass and append two more fields (after `best_cv_mean`):

```python
    kaggle_client: Any | None = None
    competition: str = ""
```

- [ ] **Step 6: Run, observe pass**

```bash
pytest tests/unit/test_handlers_submit_kaggle.py tests/unit/test_handlers_ml.py -v
```

Expected: 6 new tests pass; all existing tests pass.

- [ ] **Step 7: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/ml.py kaggle_slayer/agent/solver.py tests/unit/test_handlers_submit_kaggle.py
git commit -m "$(cat <<'EOF'
feat(agent): add submit_kaggle tool with regression-aware checkpoint

submit_kaggle(csv_path, message) routes through ctx.checkpoint_handler
with one of three triggers:
  - SUBMIT_KAGGLE_FIRST (no prior journalled submission)
  - SUBMIT_KAGGLE_REGRESSION (current best_cv_mean < previous submit's
    cv_at_submit)
  - SUBMIT_KAGGLE_NON_REGRESSION (current >= previous)

train_cv now updates ctx.best_cv_mean after each successful CV run.
submit_kaggle journals cv_at_submit alongside the tool_call entry so
future regression detection has a stable history. Path traversal is
blocked via _resolve_under.

SolverContext gains kaggle_client and competition fields. The CLI
(Task 9) wires a real KaggleClient + competition name through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wall-clock + cost-budget as Solver gate triggers

The Solver already has `max_iterations` and `time_budget_s` exit conditions. Per spec §9, wall-clock and cost budgets are checkpoint triggers, not hard exits: the user can grant an extension. This task converts those into gated decisions, and adds cost-budget tracking.

**Files:**
- Modify: `kaggle_slayer/agent/solver.py` — wall-clock + cost-budget checkpoints; new `cost_budget_usd` parameter
- Modify: `tests/unit/test_solver.py` — extend with checkpoint trigger tests

- [ ] **Step 1: Failing tests**

Append to `tests/unit/test_solver.py`:

```python
def test_solver_wall_clock_checkpoint_extends_on_approve(tmp_path):
    """When time_budget_s elapses, checkpoint fires; APPROVE extends by another budget."""
    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    # The handler approves the first wall-clock checkpoint, denies the second.
    decisions = iter([cp.Decision.APPROVE, cp.Decision.DENY])

    def prompt(_req):
        return next(decisions)

    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt
    )
    # Endless thinking-aloud responses; never call done.
    responses = [Response(text="thinking", tool_calls=[], usage=Usage(0, 0, 0)) for _ in range(50)]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=50,
        time_budget_s=0.001,  # effectively immediate
        checkpoint_handler=handler,
    )
    result = solver.solve()
    # First wall-clock checkpoint approved → continued; second denied → exit.
    assert result.status == "time_exceeded"
    # Journal must contain exactly two wall_clock_budget checkpoints.
    import json
    cp_records = [
        json.loads(l) for l in ws.run_log_path.read_text().splitlines()
        if json.loads(l).get("kind") == "checkpoint"
    ]
    wall_cp = [r for r in cp_records if r["trigger"] == "wall_clock_budget"]
    assert len(wall_cp) == 2
    assert wall_cp[0]["decision"] == "approve"
    assert wall_cp[1]["decision"] == "deny"


def test_solver_cost_budget_checkpoint(tmp_path):
    """When cost_ledger.total_for(competition) > cost_budget_usd, checkpoint fires."""
    from unittest.mock import MagicMock

    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )

    ledger = MagicMock()
    ledger.total_for = MagicMock(return_value=0.10)  # already over a $0.05 budget

    responses = [
        Response(text="thinking", tool_calls=[], usage=Usage(0, 0, 0)) for _ in range(5)
    ]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=10,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=0.05,
    )
    result = solver.solve()
    # Denied at first cost-budget check → exit with cost_budget_exceeded.
    assert result.status == "cost_budget_exceeded"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_solver.py::test_solver_wall_clock_checkpoint_extends_on_approve -v
```

Expected: fails because `Solver` doesn't accept `checkpoint_handler` / `cost_ledger` / `cost_budget_usd`.

- [ ] **Step 3: Extend `Solver.__init__` in `kaggle_slayer/agent/solver.py`**

Replace the `__init__` signature and body. New version:

```python
    def __init__(
        self,
        *,
        workspace: Workspace,
        llm_client: LLMClient,
        target_col: str = "target",
        problem_type: str = "classification",
        metric_name: str = "accuracy",
        max_iterations: int = 25,
        time_budget_s: float = 900.0,
        registry: ToolRegistry | None = None,
        checkpoint_handler: Any | None = None,
        cost_ledger: Any | None = None,
        cost_budget_usd: float | None = None,
        kaggle_client: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.time_budget_s = time_budget_s
        self.registry = registry or make_builtin_registry()
        self.journal = Journal(workspace)
        self.checkpoint_handler = checkpoint_handler
        self.cost_ledger = cost_ledger
        self.cost_budget_usd = cost_budget_usd
        self.ctx = SolverContext(
            workspace=workspace,
            journal=self.journal,
            target_col=target_col,
            problem_type=problem_type,
            metric_name=metric_name,
            checkpoint_handler=checkpoint_handler,
            kaggle_client=kaggle_client,
            competition=workspace.name,
        )
```

- [ ] **Step 4: Wire wall-clock + cost-budget checks in `solve()`**

Inside the `for iteration in range(...)` loop of `solve()`, REPLACE the existing wall-clock check:

```python
            if time.perf_counter() - started > self.time_budget_s:
                return SolveResult(status="time_exceeded", iterations=iteration - 1, summary="")
```

with this:

```python
            if time.perf_counter() - started > self.time_budget_s:
                if not self._gate_wall_clock(iteration - 1):
                    return SolveResult(status="time_exceeded", iterations=iteration - 1, summary="")
                # Approved — extend the budget by one more cycle of the original cap.
                started = time.perf_counter()

            if self._cost_budget_exceeded():
                if not self._gate_cost_budget():
                    return SolveResult(status="cost_budget_exceeded", iterations=iteration - 1, summary="")
```

Add two new methods on `Solver` (place them after `_dispatch`):

```python
    def _gate_wall_clock(self, iterations_so_far: int) -> bool:
        """Ask the checkpoint handler to extend the wall-clock budget. True = continue."""
        if self.checkpoint_handler is None:
            return False
        from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

        decision = self.checkpoint_handler.request(cp.CheckpointRequest(
            trigger=cp.CheckpointTrigger.WALL_CLOCK_BUDGET,
            action=f"extend wall-clock budget (iter {iterations_so_far})",
            evidence={"budget_s": self.time_budget_s, "iter": iterations_so_far},
        ))
        return decision in (cp.Decision.APPROVE, cp.Decision.SKIP_CHECK)

    def _cost_budget_exceeded(self) -> bool:
        if self.cost_ledger is None or self.cost_budget_usd is None:
            return False
        spent = float(self.cost_ledger.total_for(competition=self.workspace.name))
        return spent > self.cost_budget_usd

    def _gate_cost_budget(self) -> bool:
        if self.checkpoint_handler is None or self.cost_ledger is None:
            return False
        from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

        spent = float(self.cost_ledger.total_for(competition=self.workspace.name))
        decision = self.checkpoint_handler.request(cp.CheckpointRequest(
            trigger=cp.CheckpointTrigger.COST_BUDGET,
            action="cost budget exceeded — raise it?",
            evidence={"spent_usd": spent, "budget_usd": self.cost_budget_usd},
        ))
        return decision in (cp.Decision.APPROVE, cp.Decision.SKIP_CHECK)
```

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_solver.py -v
```

Expected: all Solver tests pass, including the two new ones.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/solver.py tests/unit/test_solver.py
git commit -m "$(cat <<'EOF'
feat(agent): wall-clock and cost-budget become checkpoint triggers

Solver.__init__ accepts checkpoint_handler, cost_ledger, cost_budget_usd
(all optional). When time_budget_s elapses, the loop now asks the
handler for an extension instead of hard-exiting (spec §9). When the
ledger's tracked spend for this competition crosses cost_budget_usd, a
COST_BUDGET checkpoint fires; DENY/ABORT exit as cost_budget_exceeded.

The handler stays optional so existing tests / fake-agent integration
flows keep working without a checkpoint backbone. Behaviour preserved:
if no handler is configured, wall-clock and cost-budget exit hard as
before.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Bump `result_summary` cap to match the LLM-visible cap (carry-forward for resume fidelity)

Today the journal stores `result_summary[:200]` while the LLM sees up to 8 KB. Resume (Task 8) replays from the journal — so resumed history would lose 39× of the tool-result content. Bump the journal cap to 8 KB so resume is high-fidelity.

**Files:**
- Modify: `kaggle_slayer/agent/solver.py` — change the 200-char truncation in `_dispatch` to 8000
- Modify: `tests/unit/test_solver.py` — adjust expectations

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_solver.py`:

```python
def test_solver_journals_full_capped_tool_result_for_resume_fidelity(tmp_path):
    """The run_log result_summary must store up to ~8KB so resume can
    reconstruct what the LLM actually saw on the original turn."""
    ws = _make_workspace_and_ctx(tmp_path)

    big_text = "x" * 4096  # 4 KB — well above the old 200-char cap, below the new 8 KB cap
    # Custom registry with a single handler that returns big_text
    from kaggle_slayer.agent.tools import Tool, ToolRegistry
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="returns the supplied content",
        schema={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        handler=lambda ctx, content: content,
    ))
    reg.register(Tool(
        name="done",
        description="signal finished",
        schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        handler=lambda ctx, summary: (setattr(ctx, "finished", True), setattr(ctx, "final_summary", summary))[0],
    ))

    client = _CannedClient(responses=[
        Response(text="", tool_calls=[ToolCall(id="t1", name="echo", args={"content": big_text})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="done", args={"summary": "done"})], usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5, registry=reg)
    solver.solve()

    import json
    records = [json.loads(l) for l in ws.run_log_path.read_text().splitlines()]
    echo_record = next(r for r in records if r.get("tool") == "echo")
    # Must contain the full 4 KB (or close to it), not be truncated at 200 chars.
    assert len(echo_record["result_summary"]) >= 4000
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_solver.py::test_solver_journals_full_capped_tool_result_for_resume_fidelity -v
```

Expected: assertion fails because the old code truncated to 200 chars.

- [ ] **Step 3: Change the truncation in `kaggle_slayer/agent/solver.py`**

In `Solver._dispatch`, replace:

```python
            self.journal.log_tool_call(
                tool=name,
                args=args,
                result_summary=text_result[:200],
            )
```

with:

```python
            self.journal.log_tool_call(
                tool=name,
                args=args,
                # 8 KB matches the LLM-visible cap, so resume can replay
                # exactly what the LLM originally saw.
                result_summary=text_result[:8000],
            )
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_solver.py -v
```

Expected: new test passes; existing tests still pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/solver.py tests/unit/test_solver.py
git commit -m "$(cat <<'EOF'
fix(solver): bump journal result_summary cap to 8 KB for resume fidelity

The journal previously truncated tool results to 200 chars, while the
LLM saw up to 8 KB on each turn. Resume (Task 8) replays history from
the journal — so resumed runs would lose ~39× of the actual context.
Align the two caps at 8 KB. Storage cost is bounded by the same cap
that bounds the live conversation, so no new failure modes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `resume.rebuild_conversation` — replay the journal as a message list

Walks `run_log.jsonl` and emits a `list[Message]` matching what the Solver originally sent. For each tool_call: a `model` message with the function_call and a `tool` message with the result. For each tool_error: same pattern but with the error text as the result.

**Files:**
- Modify: `kaggle_slayer/harness/resume.py` — add `rebuild_conversation`, `ResumeError`
- Create: `tests/unit/test_resume_rebuild.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_resume_rebuild.py`:

```python
"""Tests for resume.rebuild_conversation."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.agent.llm_client import Message
from kaggle_slayer.harness import resume
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_rebuild_empty_journal_returns_empty_list(ws):
    """No prior runs → empty resume list."""
    assert resume.rebuild_conversation(ws) == []


def test_rebuild_includes_one_model_one_tool_per_call(ws):
    """Each tool_call record becomes a model(function_call) + tool(function_response)."""
    j = Journal(ws)
    j.log_tool_call(tool="read_context", args={}, result_summary="# Comp\nMetric: accuracy")
    j.log_tool_call(tool="write_file", args={"path": "agent/fe.py", "content": "..."},
                    result_summary="wrote 3 bytes")
    msgs = resume.rebuild_conversation(ws)
    # 2 tool calls → 4 messages (2 × (model + tool))
    assert len(msgs) == 4
    assert msgs[0].role == "model"
    assert msgs[0].tool_calls[0].name == "read_context"
    assert msgs[1].role == "tool"
    assert "Comp" in msgs[1].content
    assert msgs[2].role == "model"
    assert msgs[2].tool_calls[0].name == "write_file"
    assert msgs[2].tool_calls[0].args == {"path": "agent/fe.py", "content": "..."}
    assert msgs[3].role == "tool"
    assert "3 bytes" in msgs[3].content


def test_rebuild_handles_tool_errors_as_model_call_plus_error_response(ws):
    """tool_error records also emit model+tool pairs (result is the error message)."""
    j = Journal(ws)
    j.log_tool_error(tool="write_file", args={"path": "context.md", "content": "x"},
                     error="ToolError: path 'context.md' is protected")
    msgs = resume.rebuild_conversation(ws)
    assert len(msgs) == 2
    assert msgs[0].role == "model"
    assert msgs[0].tool_calls[0].name == "write_file"
    assert msgs[1].role == "tool"
    assert "protected" in msgs[1].content


def test_rebuild_skips_checkpoint_records(ws):
    """checkpoint records are journalled but not part of the LLM conversation."""
    j = Journal(ws)
    j._append(ws.run_log_path, {  # noqa: SLF001
        "ts": "2026-05-15", "kind": "checkpoint", "trigger": "set_metric",
        "action": "change metric", "evidence": {}, "decision": "approve",
    })
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.82")
    msgs = resume.rebuild_conversation(ws)
    # Only the train_cv call should produce messages — checkpoint stays in the log.
    assert len(msgs) == 2
    assert msgs[0].tool_calls[0].name == "train_cv"


def test_rebuild_raises_when_done_already_called(ws):
    """A workspace whose last tool_call was 'done' has nothing to resume."""
    j = Journal(ws)
    j.log_tool_call(tool="write_file", args={"path": "x", "content": "y"}, result_summary="ok")
    j.log_tool_call(tool="done", args={"summary": "all done"}, result_summary="ack")
    with pytest.raises(resume.ResumeError, match="already finished"):
        resume.rebuild_conversation(ws)


def test_rebuild_handles_missing_log_file(ws):
    """A workspace without run_log.jsonl returns an empty list."""
    # Don't write anything; run_log_path doesn't exist
    assert not ws.run_log_path.exists()
    assert resume.rebuild_conversation(ws) == []
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_resume_rebuild.py -v
```

Expected: `AttributeError: module 'kaggle_slayer.harness.resume' has no attribute 'rebuild_conversation'`.

- [ ] **Step 3: Add `rebuild_conversation` and `ResumeError` to `kaggle_slayer/harness/resume.py`**

Append to the existing file:

```python
import json as _json

from kaggle_slayer.agent.llm_client import Message, ToolCall


class ResumeError(Exception):
    """Raised when the journal is in a state from which we cannot resume."""


def rebuild_conversation(workspace: Workspace) -> list[Message]:
    """Replay run_log.jsonl as the Message history the Solver originally sent.

    For each tool_call / tool_error record:
      - emit a model(role) Message with that call's tool_calls=[ToolCall(...)]
      - emit a tool(role) Message with the result (or the error string)

    checkpoint records are ignored (they're not part of the LLM conversation).
    Raises ResumeError if the last tool_call was 'done' (workspace finished).
    """
    j = Journal(workspace)
    records = list(j.iter_records())
    if not records:
        return []

    # Look at the last tool_call (skipping checkpoint records) — if it's `done`,
    # the run is finished and there's nothing to resume.
    last_tool_record = next(
        (r for r in reversed(records) if r.get("kind") in ("tool_call", "tool_error")),
        None,
    )
    if last_tool_record is not None and last_tool_record.get("tool") == "done":
        raise ResumeError(
            "workspace already finished (last tool call was 'done'); "
            "delete run_log.jsonl to start fresh"
        )

    messages: list[Message] = []
    for rec in records:
        kind = rec.get("kind")
        if kind == "tool_call":
            tool_name = rec.get("tool", "unknown")
            args = rec.get("args", {})
            result = rec.get("result_summary", "")
            tc = ToolCall(id=f"resume_{len(messages)}", name=tool_name, args=args)
            messages.append(Message(role="model", content="", tool_calls=[tc]))
            payload = _json.dumps({"tool": tool_name, "result": result})
            messages.append(Message(role="tool", content=payload))
        elif kind == "tool_error":
            tool_name = rec.get("tool", "unknown")
            args = rec.get("args", {})
            error = rec.get("error", "")
            tc = ToolCall(id=f"resume_{len(messages)}", name=tool_name, args=args)
            messages.append(Message(role="model", content="", tool_calls=[tc]))
            payload = _json.dumps({"tool": tool_name, "result": error})
            messages.append(Message(role="tool", content=payload))
        # kind=='checkpoint' is silently skipped
    return messages
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_resume_rebuild.py -v
```

Expected: 6 passes.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/resume.py tests/unit/test_resume_rebuild.py
git commit -m "$(cat <<'EOF'
feat(harness): add resume.rebuild_conversation — replay journal as messages

rebuild_conversation(workspace) walks run_log.jsonl and emits a
list[Message] mirroring what the Solver originally sent: one model
Message with a ToolCall per tool_call/tool_error record, followed by a
tool-role Message carrying the result (or error text). checkpoint
records are skipped — they're journalled but not part of the LLM
conversation.

Raises ResumeError if the last tool call was 'done' (no work to
resume). Empty / missing journal returns []. The reconstructed list is
high-fidelity now that Task 7 bumped result_summary to the LLM-visible
8 KB cap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: CLI `--resume`, `--cost-budget`, `--auto-approve` flags

The CLI grows three new flags. `--resume` builds the prior conversation via `rebuild_conversation` and seeds the Solver. `--cost-budget` plumbs a number to the Solver. `--auto-approve` chooses the checkpoint handler mode (interactive by default, `safe` for the auto-approve subset, `all` for tests).

**Files:**
- Modify: `kaggle_slayer/cli.py`
- Modify: `kaggle_slayer/agent/solver.py` — `solve()` accepts `resume_from`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Extend `Solver.solve()` to accept seeded history**

In `kaggle_slayer/agent/solver.py`, change the `solve()` signature from `def solve(self) -> SolveResult:` to:

```python
    def solve(self, *, resume_from: list[Message] | None = None) -> SolveResult:
```

In the body, replace the existing `messages: list[Message] = [...]` block with:

```python
        if resume_from:
            messages: list[Message] = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=context_md),
                *resume_from,
            ]
        else:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=context_md),
            ]
```

(So a resumed run includes system + context + every prior tool turn. Tests verify the seeded messages reach the LLM call.)

- [ ] **Step 2: Failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_parses_resume_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--resume", "--target", "y"])
    assert args.resume is True


def test_cli_parses_cost_budget_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--cost-budget", "0.25", "--target", "y"])
    assert args.cost_budget == 0.25


def test_cli_parses_auto_approve_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--auto-approve", "safe", "--target", "y"])
    assert args.auto_approve == "safe"


def test_cli_resume_passes_rebuilt_history_to_solver(tmp_path):
    """When --resume is set and run_log.jsonl has prior tool calls, those
    messages are passed via solve(resume_from=...)."""
    import pandas as pd
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)
    # Seed a journal entry that the resume should pick up
    from kaggle_slayer.harness.workspace import Workspace
    from kaggle_slayer.harness.journal import Journal
    ws = Workspace.create(root=comp_path)
    Journal(ws).log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="ok")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--resume"])

    # solve() was called with resume_from=<rebuilt history>
    call_kwargs = mock_solver.solve.call_args.kwargs
    assert "resume_from" in call_kwargs
    assert call_kwargs["resume_from"] is not None
    assert len(call_kwargs["resume_from"]) >= 2  # at least one model + tool message
```

- [ ] **Step 3: Run, observe failure**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: `--resume`, `--cost-budget`, `--auto-approve` flags absent.

- [ ] **Step 4: Add the flags and wire them in `kaggle_slayer/cli.py`**

Find `_parse_args` and append before `return p.parse_args(argv)`:

```python
    p.add_argument("--resume", action="store_true",
                   help="Resume from run_log.jsonl (rebuilds conversation history)")
    p.add_argument("--cost-budget", type=float, default=None,
                   help="USD cost cap; checkpoint fires when exceeded")
    p.add_argument("--auto-approve", choices=["off", "safe", "all"], default="off",
                   help="Checkpoint mode: off=interactive, safe=auto-approve non-regression submits only, all=auto-approve everything (tests only)")
```

Find the `run()` function. Just before the existing `solver = Solver(...)` block, add:

```python
    # Checkpoint handler
    from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415
    if args.auto_approve == "safe":
        handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=Journal(workspace))
    elif args.auto_approve == "all":
        handler = cp.CheckpointHandler(
            mode=cp.HandlerMode.STUB,
            journal=Journal(workspace),
            stub_decision=cp.Decision.APPROVE,
        )
    else:
        handler = cp.CheckpointHandler(mode=cp.HandlerMode.INTERACTIVE, journal=Journal(workspace))

    # Resume?
    resume_from = None
    if args.resume:
        from kaggle_slayer.harness import resume as resume_mod  # noqa: PLC0415
        try:
            resume_from = resume_mod.rebuild_conversation(workspace)
        except resume_mod.ResumeError as e:
            print(f"resume failed: {e}", file=sys.stderr)
            return 3
```

Add to the imports at the top:

```python
from kaggle_slayer.harness.journal import Journal
```

And modify the `Solver(...)` instantiation to pass the new args:

```python
    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col=args.target or "target",
        problem_type=args.problem_type,
        metric_name=args.metric,
        max_iterations=args.max_iterations,
        time_budget_s=args.time_budget_s,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=args.cost_budget,
        kaggle_client=KaggleClient(),
    )
    result = solver.solve(resume_from=resume_from)
```

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: new tests pass; existing tests still pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/cli.py kaggle_slayer/agent/solver.py tests/unit/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): add --resume, --cost-budget, --auto-approve flags

--resume                rebuild the conversation from run_log.jsonl
                        before starting the Solver loop
--cost-budget <USD>     hand the value to Solver(cost_budget_usd=...);
                        Solver fires a COST_BUDGET checkpoint when the
                        ledger's spend for this competition exceeds it
--auto-approve {off,    pick the CheckpointHandler mode:
   safe, all}             off  = interactive rich CLI prompt (default)
                          safe = auto-approve non-regression submits
                          all  = auto-approve every checkpoint (tests)

Solver.solve() now takes an optional resume_from=list[Message] that's
inserted between the system+context messages and the first LLM call.
Resume failures (e.g., last call was 'done') exit non-zero with a
human-readable message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Register the new tools + update the system prompt

Wire `run_python`, `set_metric`, `submit_kaggle`, and `request_human_approval` into `make_builtin_registry`. Update `system.md` to describe them.

**Files:**
- Modify: `kaggle_slayer/agent/handlers/__init__.py` — register four new tools
- Modify: `kaggle_slayer/agent/handlers/ml.py` — add `request_human_approval` function
- Modify: `kaggle_slayer/agent/prompts/system.md`
- Modify: `tests/unit/test_builtin_registry.py` — expected tool set grows from 9 to 13

- [ ] **Step 1: Failing test**

Find the existing `test_builtin_registry_has_expected_tools` in `tests/unit/test_builtin_registry.py`. Change it to:

```python
def test_builtin_registry_has_expected_tools(ctx):
    reg = make_builtin_registry()
    expected = {
        "read_context", "read_file", "write_file", "sample_rows",
        "take_note", "set_cv", "train_cv", "submit_local", "done",
        "run_python", "set_metric", "submit_kaggle", "request_human_approval",
    }
    assert set(reg.names()) == expected
```

Also update `test_builtin_registry_function_declarations_format`:

```python
def test_builtin_registry_function_declarations_format(ctx):
    reg = make_builtin_registry()
    decls = reg.to_function_declarations()
    assert len(decls) == 13
    for d in decls:
        assert d["name"]
        assert d["description"]
        assert d["parameters"]["type"] == "object"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_builtin_registry.py -v
```

Expected: assertion fails because the registry currently has 9 tools.

- [ ] **Step 3: Add `request_human_approval` to `kaggle_slayer/agent/handlers/ml.py`**

Append:

```python
def request_human_approval(ctx: Any, *, action: str, evidence_json: str = "{}") -> str:
    """Agent-initiated checkpoint: pause and ask the human to weigh in."""
    handler = getattr(ctx, "checkpoint_handler", None)
    if handler is None:
        raise ToolError("request_human_approval needs a checkpoint handler on the context")

    import json as _json  # noqa: PLC0415
    from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

    try:
        evidence = _json.loads(evidence_json) if evidence_json else {}
        if not isinstance(evidence, dict):
            evidence = {"value": evidence}
    except _json.JSONDecodeError:
        evidence = {"raw_evidence": evidence_json}

    decision = handler.request(cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.AGENT_INITIATED,
        action=action,
        evidence=evidence,
    ))
    return f"decision={decision.value}"
```

- [ ] **Step 4: Register the four new tools in `kaggle_slayer/agent/handlers/__init__.py`**

Add imports at the top (next to the existing `from kaggle_slayer.agent.handlers import files as fh, ml as ml_h`):

```python
from kaggle_slayer.agent.handlers import python as ph_python  # noqa: I001 — keep grouped with handlers
```

(Yes, alphabetical: `files`, `ml`, then `python`. Reshuffle imports so ruff is happy.)

In `make_builtin_registry()`, append before `return reg`:

```python
    reg.register(Tool(
        name="run_python",
        description=(
            "Run a Python snippet in a sandboxed subprocess (resource-limited, "
            "cwd=workspace root). Use for plotting, peeks, quick debug. Returns "
            "stdout/stderr/returncode as a string. Do NOT use for CV — train_cv "
            "is the only path to a valid CV score."
        ),
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600, "default": 60},
                "memory_mb": {"type": "integer", "minimum": 64, "maximum": 16384, "default": 2048},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=ph_python.run_python,
    ))
    reg.register(Tool(
        name="set_metric",
        description=(
            "Change the scoring metric. Checkpoint-gated. Use when the parsed "
            "metric is wrong (e.g., comp uses weighted F1 but the harness picked "
            "accuracy). Pick from accuracy/auc/logloss/rmse/mae/r2."
        ),
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=ml_h.set_metric,
    ))
    reg.register(Tool(
        name="submit_kaggle",
        description=(
            "Push a submission CSV to Kaggle. Checkpoint-gated (always for the "
            "first submission of a comp; on score regression for subsequent ones). "
            "csv_path is workspace-relative."
        ),
        schema={
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["csv_path", "message"],
            "additionalProperties": False,
        },
        handler=ml_h.submit_kaggle,
    ))
    reg.register(Tool(
        name="request_human_approval",
        description=(
            "Pause and ask the human for explicit approval before proceeding. "
            "Use when you're about to take an action whose stakes you can't fully "
            "judge (e.g., a non-obvious metric override, an unusual leak risk). "
            "evidence_json is a JSON-encoded dict of context the human should see."
        ),
        schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "evidence_json": {"type": "string", "default": "{}"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=ml_h.request_human_approval,
    ))
```

- [ ] **Step 5: Update `kaggle_slayer/agent/prompts/system.md`**

Find the "## Workflow" section's numbered list. After step 7 (`Call done(summary=...) to finish.`), append:

```markdown

## Additional tools

- `run_python(code, timeout_s, memory_mb)` — sandboxed Python (plotting, peeks, debug). NOT for CV.
- `set_metric(name)` — change the scoring metric. Always asks the human first.
- `submit_kaggle(csv_path, message)` — push a submission CSV to Kaggle. Always asks the human on the first submission, and on any score regression.
- `request_human_approval(action, evidence_json)` — pause and ask the human when you're uncertain.
```

Also add a one-liner inside the Hard rules section near the top:

```markdown
- DON'T call `submit_kaggle` to "test" the API — every submission counts against the daily limit.
```

- [ ] **Step 6: Run, observe pass**

```bash
pytest tests/unit/test_builtin_registry.py tests/unit/test_handlers_set_metric.py tests/unit/test_handlers_submit_kaggle.py -v
```

Expected: all pass; registry size = 13.

- [ ] **Step 7: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/__init__.py kaggle_slayer/agent/handlers/ml.py kaggle_slayer/agent/prompts/system.md tests/unit/test_builtin_registry.py
git commit -m "$(cat <<'EOF'
feat(agent): register run_python + set_metric + submit_kaggle + request_human_approval

The builtin registry grows from 9 to 13 tools. The four new tools all
have additionalProperties:false on their schemas. set_metric and
submit_kaggle route through ctx.checkpoint_handler; request_human_approval
is the agent's escape hatch when it's uncertain.

system.md now documents all 13 tools and adds a 'don't test the kaggle
API' hard-rule warning (every submission counts toward the daily cap).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Integration test — checkpoint approval / denial flow

Scripted FakeLLMClient drives the Solver through a sequence that triggers `set_metric` (always-gated) and `submit_kaggle` (first-time gated). One run with the handler approving; one with the handler denying.

**Files:**
- Create: `tests/integration/test_checkpoint_flow.py`

- [ ] **Step 1: Failing test**

Create `tests/integration/test_checkpoint_flow.py`:

```python
"""Integration: Solver loop drives set_metric and submit_kaggle checkpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from tests.fixtures.synthetic_comp import make_synthetic_comp


pytestmark = pytest.mark.integration


_FE_CODE = '''
import pandas as pd

class _PT:
    def __init__(self, cols, means):
        self.cols, self.means = cols, means
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            if c in df.columns:
                out[c] = df[c].fillna(self.means.get(c, 0.0))
        if "id" in out.columns:
            out = out.drop(columns=["id"])
        return out

def fit_feature_transformer(train_df, target_col):
    cols = [c for c in train_df.columns if c not in (target_col, "id") and train_df[c].dtype.kind in "fiub"]
    means = {c: float(train_df[c].mean()) for c in cols}
    return _PT(cols, means)
'''

_MODEL_CODE = '''
from sklearn.linear_model import LogisticRegression
def fit_model(X_train, y_train, problem_type, metric_name):
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(X_train, y_train)
    return m
'''


class _Scripted:
    def __init__(self, responses):
        self._r, self._i = list(responses), 0
        self.captured = []
    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._r[self._i]
        self._i += 1
        return r


def _build_solver(workspace, client, handler):
    fake_kaggle = MagicMock()
    return Solver(
        workspace=workspace,
        llm_client=client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=15,
        checkpoint_handler=handler,
        kaggle_client=fake_kaggle,
    ), fake_kaggle


def test_set_metric_approved_then_submit_kaggle_approved(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE)
    client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="set_metric", args={"name": "auc"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="write_file", args={"path": "agent/fe.py", "content": _FE_CODE})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t3", name="write_file", args={"path": "agent/model.py", "content": _MODEL_CODE})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t4", name="train_cv", args={})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t5", name="submit_local", args={"label": "v1"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t6", name="submit_kaggle", args={"csv_path": None, "message": "v1"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t7", name="done", args={"summary": "fin"})], usage=Usage(0, 0, 0)),
    ])
    solver, fake_kaggle = _build_solver(workspace, client, handler)

    # Find the submit_local output to feed into submit_kaggle's csv_path.
    # Easier: patch the scripted sequence after the solver writes submissions/v1.csv.
    # Trick: pre-write the submission file so submit_kaggle's path resolves.
    (workspace.submissions_dir / "manual.csv").write_text("id,target\n1,0\n")
    client._r[5] = Response(text="", tool_calls=[ToolCall(
        id="t6", name="submit_kaggle",
        args={"csv_path": "submissions/manual.csv", "message": "v1"},
    )], usage=Usage(0, 0, 0))

    result = solver.solve()
    assert result.status == "done"
    fake_kaggle.submit.assert_called_once()

    cp_records = [
        json.loads(l) for l in workspace.run_log_path.read_text().splitlines()
        if json.loads(l).get("kind") == "checkpoint"
    ]
    triggers = [r["trigger"] for r in cp_records]
    assert "set_metric" in triggers
    assert any(t.startswith("submit_kaggle") for t in triggers)
    # Every checkpoint was approved
    assert all(r["decision"] == "approve" for r in cp_records)


def test_set_metric_denied_keeps_original_metric(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY)
    client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="set_metric", args={"name": "auc"})], usage=Usage(0, 0, 0)),
        # After being denied, the agent gives up and calls done
        Response(text="", tool_calls=[ToolCall(id="t2", name="done", args={"summary": "blocked"})], usage=Usage(0, 0, 0)),
    ])
    solver, fake_kaggle = _build_solver(workspace, client, handler)
    result = solver.solve()

    # Metric stays as the original 'accuracy'
    assert solver.ctx.metric_name == "accuracy"
    # The tool dispatch fed a ToolError back to the model
    tool_results = [m for m in client.captured[1] if m.role == "tool"]
    assert any("denied" in m.content.lower() for m in tool_results)
    # Kaggle was never touched
    fake_kaggle.submit.assert_not_called()
    assert result.status == "done"
```

- [ ] **Step 2: Run, observe pass**

```bash
pytest tests/integration/test_checkpoint_flow.py -v -m integration
```

Expected: 2 passes.

- [ ] **Step 3: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/integration/test_checkpoint_flow.py
git commit -m "$(cat <<'EOF'
test(integration): scripted checkpoint approval + denial flow

Drives the Solver through set_metric and submit_kaggle in two flavours:
  - all-approve: handler returns APPROVE; submit_kaggle reaches the fake
    kaggle_client; both checkpoints land in run_log.jsonl
  - all-deny: set_metric is denied; the tool-error message is fed back
    to the LLM (so the agent sees what happened); the kaggle_client is
    never touched

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Integration test — resume mid-run

A workspace gets partway through a solve, then a fresh Solver is created with `resume_from=rebuild_conversation(workspace)`. Verify the resumed run picks up where it left off and the LLM sees the prior tool history on the first call.

**Files:**
- Create: `tests/integration/test_resume_flow.py`

- [ ] **Step 1: Failing test**

Create `tests/integration/test_resume_flow.py`:

```python
"""Integration: aborted Solver run can be resumed via rebuild_conversation."""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import resume
from tests.fixtures.synthetic_comp import make_synthetic_comp


pytestmark = pytest.mark.integration


class _Scripted:
    def __init__(self, responses):
        self._r, self._i = list(responses), 0
        self.captured = []
    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._r[self._i]
        self._i += 1
        return r


def test_resume_picks_up_after_three_tool_calls(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")

    # Phase 1: run three steps, then halt at max_iterations=3.
    phase1_responses = [
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t1", name="take_note",
                                      args={"category": "observation", "content": "binary target"})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t2", name="sample_rows",
                                      args={"table": "train", "n": 5})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t3", name="take_note",
                                      args={"category": "decision", "content": "use logistic regression"})],
                 usage=Usage(0, 0, 0)),
    ]
    phase1_client = _Scripted(phase1_responses)
    phase1 = Solver(
        workspace=workspace, llm_client=phase1_client,
        target_col="Survived", max_iterations=3,
    )
    r1 = phase1.solve()
    # Loop exited at max_iterations (no done call yet)
    assert r1.status == "max_iterations"

    # Phase 2: rebuild history, then a fresh Solver finishes the comp.
    history = resume.rebuild_conversation(workspace)
    # 3 prior tool calls → 6 messages (model + tool per call)
    assert len(history) == 6

    phase2_responses = [
        Response(text="",
                 tool_calls=[ToolCall(id="p2_t1", name="done",
                                      args={"summary": "resumed and finished"})],
                 usage=Usage(0, 0, 0)),
    ]
    phase2_client = _Scripted(phase2_responses)
    phase2 = Solver(
        workspace=workspace, llm_client=phase2_client,
        target_col="Survived", max_iterations=5,
    )
    r2 = phase2.solve(resume_from=history)
    assert r2.status == "done"
    assert "resumed" in r2.summary

    # The first LLM call in phase 2 must have included the 6 resumed messages.
    first_call_msgs = phase2_client.captured[0]
    # system + user(context) + 6 resumed = 8 minimum
    assert len(first_call_msgs) >= 8
    # The 3rd message (after system+user) should be a model-role with take_note
    assert first_call_msgs[2].role == "model"
    assert first_call_msgs[2].tool_calls[0].name == "take_note"


def test_resume_raises_when_done_already_called(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    phase1_client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="done",
                                               args={"summary": "first run done"})],
                 usage=Usage(0, 0, 0)),
    ])
    Solver(workspace=workspace, llm_client=phase1_client, target_col="Survived").solve()

    with pytest.raises(resume.ResumeError, match="already finished"):
        resume.rebuild_conversation(workspace)
```

- [ ] **Step 2: Run, observe pass**

```bash
pytest tests/integration/test_resume_flow.py -v -m integration
```

Expected: 2 passes.

- [ ] **Step 3: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/integration/test_resume_flow.py
git commit -m "$(cat <<'EOF'
test(integration): mid-run abort + resume picks up where it left off

Phase 1 runs three tool calls then halts at max_iterations=3 (no done).
Phase 2 calls rebuild_conversation(workspace) (returns 6 messages: model
+ tool per prior call), constructs a fresh Solver, and calls
solve(resume_from=history). The first LLM call in phase 2 receives
system + user(context) + 6 resumed messages, then the scripted client
calls done → status=done in 1 iteration.

A second test asserts rebuild_conversation raises ResumeError when the
journal's last tool call was 'done' (nothing to resume).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Real-Gemini acceptance — solve with a checkpoint + mocked submit_kaggle (slow tier)

Final gate: real Gemini solves a synthetic comp, hits a checkpoint (we use `auto-approve=safe`), and reaches `submit_kaggle`, which calls a mocked `KaggleClient.submit`. The CSV is real; Kaggle never sees it.

**Files:**
- Create: `tests/integration/test_solver_real_gemini_checkpoint.py`

- [ ] **Step 1: Write the slow E2E test**

```python
"""Real-Gemini E2E acceptance — gated submission flow. Slow tier, opt-in.

Uses gemini-2.5-flash + AUTO_SAFE checkpoint mode + a mocked KaggleClient.
The CSV is generated for real (the model trains; submit_local writes it),
but no actual Kaggle upload happens — kaggle_client.submit is a MagicMock.

Cost: ~$0.005-0.02 per run.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from tests.fixtures.synthetic_comp import make_synthetic_comp

load_dotenv()

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env")
    return key


def test_real_gemini_completes_with_gated_submit_kaggle(tmp_path, gemini_key):
    """Real Gemini solves the synthetic comp, attempts submit_kaggle, the
    AUTO_SAFE checkpoint approves (because there's no prior submission to
    regress against — wait, AUTO_SAFE denies first submissions; use STUB
    auto-approve here instead). Mocked KaggleClient.submit is called."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    fake_kaggle = MagicMock()

    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    llm = GeminiClient(
        api_key=gemini_key, ledger=ledger, competition="synthetic-checkpoint-e2e",
        default_model="gemini-2.5-flash", retry_max=4, retry_base_delay_s=20.0,
    )
    solver = Solver(
        workspace=workspace, llm_client=llm,
        target_col="Survived", problem_type="classification", metric_name="accuracy",
        max_iterations=25, time_budget_s=900.0,
        checkpoint_handler=handler, kaggle_client=fake_kaggle,
    )

    # Nudge the agent toward calling submit_kaggle after submit_local. Patch
    # `kaggle_slayer.agent.solver.load_system_prompt` (the binding the Solver
    # actually references) — patching `prompts.load_system_prompt` would NOT
    # reach it because solver.py did `from ... import load_system_prompt` at
    # import time and has its own reference.
    from unittest.mock import patch
    from kaggle_slayer.agent import prompts as p_mod
    original_loader = p_mod.load_system_prompt

    def loader_with_kaggle():
        return original_loader() + (
            "\n\n## Extra instruction for this run\n"
            "After submit_local succeeds, you MUST call submit_kaggle "
            "with csv_path pointing at the file submit_local just wrote, "
            "and a 1-line message. Then call done."
        )

    with patch("kaggle_slayer.agent.solver.load_system_prompt", loader_with_kaggle):
        result = solver.solve()

    if result.status != "done" and workspace.run_log_path.exists():
        print("\n--- run_log.jsonl ---")
        print(workspace.run_log_path.read_text())

    assert result.status == "done", (
        f"status={result.status} iters={result.iterations} summary={result.summary!r}"
    )
    fake_kaggle.submit.assert_called()  # at least one submit attempt reached the gate

    # At least one submit_kaggle checkpoint was journalled
    import json
    cp_records = [
        json.loads(l) for l in workspace.run_log_path.read_text().splitlines()
        if json.loads(l).get("kind") == "checkpoint"
    ]
    submit_cp = [r for r in cp_records if r["trigger"].startswith("submit_kaggle")]
    assert submit_cp, "no submit_kaggle checkpoint recorded"
    print(
        f"\nDONE iter={result.iterations} "
        f"cost=${ledger.total_for(competition='synthetic-checkpoint-e2e'):.4f}"
    )
```

- [ ] **Step 2: Confirm pytest collects but does not run**

```bash
pytest --collect-only tests/integration/test_solver_real_gemini_checkpoint.py 2>&1 | head -10
pytest -m "not slow" tests/integration/ --collect-only 2>&1 | tail -5
```

Expected: the new test appears in `--collect-only` but is excluded from `-m "not slow"`.

- [ ] **Step 3: Lint + mypy**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_solver_real_gemini_checkpoint.py
git commit -m "$(cat <<'EOF'
test: real-Gemini E2E with gated submit_kaggle (slow tier)

Real gemini-2.5-flash solves the synthetic micro-comp, attempts
submit_kaggle, the STUB-approve checkpoint approves, the (mocked)
KaggleClient.submit fires. Asserts: status=done, kaggle.submit called
at least once, at least one submit_kaggle_* checkpoint journalled.

The system prompt is patched at runtime to nudge the agent toward
calling submit_kaggle after submit_local — without that, Flash often
stops after submit_local on a 'done enough' inference.

Cost: ~$0.005-0.02 per run. Slow tier, opt-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Week 4 acceptance summary

After all 13 tasks:

- ✅ `kaggle_slayer/harness/sandbox.py:run_subprocess` — POSIX-resource-limited subprocess execution
- ✅ `kaggle_slayer/agent/handlers/python.py:run_python` — sandboxed escape hatch
- ✅ `kaggle_slayer/harness/checkpoints.py` — typed gate, four modes, journalled decisions
- ✅ `set_metric` and `submit_kaggle` (regression-aware) handlers wired into the gate
- ✅ Wall-clock and cost-budget exit conditions converted to gate triggers in the Solver
- ✅ `harness/resume.py:rebuild_conversation` — replays journal as Message history
- ✅ CLI `--resume`, `--cost-budget`, `--auto-approve` flags
- ✅ Builtin registry grows from 9 to 13 tools (run_python, set_metric, submit_kaggle, request_human_approval added)
- ✅ Two integration tests: scripted checkpoint flow, scripted resume mid-run
- ✅ Slow-tier real-Gemini acceptance: full Solver loop hits a checkpoint and a (mocked) submit
- ✅ mypy strict and ruff clean on `harness/` + `agent/`; coverage on new code ≥ 85%

**Week 5 starts with:** telemetry (cost ledger surfacing, OTel tracing, CV↔LB calibration logger), redesigned Streamlit dashboard (portfolio / comp-detail / cross-comp pages), chaos-tier tests (random tool-call failures at 5% rate).
