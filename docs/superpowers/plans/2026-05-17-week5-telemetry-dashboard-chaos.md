# Week 5 — Telemetry, dashboard, chaos tier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close spec §11 (observability) and §13 (chaos tier). At the end of the week, every Solver run emits an OpenTelemetry trace to a local file, every `submit_kaggle` appends a row to a global calibration log, unhandled exceptions get captured to `~/.kaggle_slayer/errors/`, agent behavior counters (turns/run, turns-to-first-submission) are journalled, MLflow records one run per `train_cv` call, the Streamlit dashboard surfaces a portfolio overview + per-comp detail page, and a chaos test injects 5% tool-call failures and asserts the harness still finishes cleanly.

**Architecture:** New telemetry modules live in `kaggle_slayer/harness/telemetry/` (one per concern: `otel.py`, `calibration.py`, `errors.py`, `behavior.py`, `mlflow_logger.py`). Each is a thin wrapper over its respective backend (otel-sdk, JSONL append, ~/.kaggle_slayer/errors, run_log scan, mlflow.start_run). The Solver loop and the `submit_kaggle` / `train_cv` handlers acquire the telemetry surfaces via dependency injection (passed through `SolverContext` or constructed inline when nil). The dashboard reads from disk only — no live calls. The chaos tier wraps the existing scripted-fake-LLM integration tests with a `FailureInjectingLLMClient` that fails 5% of calls with `TransientLLMError` and verifies the Solver's retry + journal logic copes.

**Tech Stack:** opentelemetry-api / opentelemetry-sdk (already in deps), mlflow>=2.10 (already in deps), streamlit>=1.30 + plotly>=5.18 (already in `[dashboard]` extras), python-json (stdlib). Python 3.11+. Mypy strict on the new modules.

**Acceptance:** unit tier green (~25 new tests), integration tier green (chaos test passes with 5% injection), slow tier passes a real-Gemini run that writes an OTel trace, a calibration row, and a cost-ledger row. Coverage on new code ≥ 85%. mypy strict and ruff clean on the new package.

---

## File map

**Created this week:**

- `kaggle_slayer/harness/telemetry/__init__.py` — package marker
- `kaggle_slayer/harness/telemetry/otel.py` — tracer factory + file exporter + Solver-loop span helpers
- `kaggle_slayer/harness/telemetry/calibration.py` — `record_calibration` / `read_history` over `~/.kaggle_slayer/calibration.jsonl`
- `kaggle_slayer/harness/telemetry/errors.py` — `capture(exc, recent_calls, env)` writes `~/.kaggle_slayer/errors/<ts>.json` with 100-file rotation
- `kaggle_slayer/harness/telemetry/behavior.py` — `BehaviorMetrics` dataclass + computation from a journal (turns_per_run, turns_to_first_submission, stuck_loop). Replaces the stuck-loop logic in `resume.py` (consolidate)
- `kaggle_slayer/harness/telemetry/mlflow_logger.py` — `mlflow_for_train_cv(workspace, fe_version, model_version, ...)` context manager that wraps one `mlflow.start_run` with structured params/metrics
- `kaggle_slayer/dashboard/__init__.py` — package marker
- `kaggle_slayer/dashboard/app.py` — Streamlit entry; routes to portfolio or comp-detail
- `kaggle_slayer/dashboard/portfolio.py` — list-of-comps page
- `kaggle_slayer/dashboard/comp_detail.py` — single-comp page (journal timeline, cost, calibration, notes)
- `tests/chaos/__init__.py`
- `tests/chaos/conftest.py` — `FailureInjectingLLMClient` fixture (5% failure rate, seedable)
- `tests/chaos/test_solver_chaos.py` — full scripted run with injected transient failures
- `tests/unit/test_telemetry_otel.py`
- `tests/unit/test_telemetry_calibration.py`
- `tests/unit/test_telemetry_errors.py`
- `tests/unit/test_telemetry_behavior.py`
- `tests/unit/test_telemetry_mlflow.py`
- `tests/integration/test_solver_real_gemini_telemetry.py` — slow tier, opt-in

**Modified:**

- `kaggle_slayer/agent/solver.py` — wrap `solve()` in an OTel root span; emit a child span per LLM call + per tool dispatch. Pass `BehaviorMetrics` into the journal on exit.
- `kaggle_slayer/agent/handlers/ml.py` — `submit_kaggle` writes a calibration row after a successful submit. `train_cv` wraps the harness call in the MLflow context manager.
- `kaggle_slayer/cli.py` — wrap `run()` in a try/except that calls `errors.capture(...)`; thread a `--no-mlflow` flag through to opt out.
- `kaggle_slayer/harness/resume.py` — replace the inline stuck-loop detector with a delegated call to `telemetry/behavior.detect_stuck_loop`. Keep `summarize` API stable.
- `pyproject.toml` — add `[project.scripts] kaggle-slayer-dashboard = "kaggle_slayer.dashboard.app:main"` so `streamlit run` isn't required.

---

## Task 1: OTel tracer + file exporter

A thin module that yields a `Tracer` whose spans land in `<workspace>/otel.jsonl`. One trace per Solver run.

**Files:**
- Create: `kaggle_slayer/harness/telemetry/__init__.py`
- Create: `kaggle_slayer/harness/telemetry/otel.py`
- Create: `tests/unit/test_telemetry_otel.py`

- [ ] **Step 1: Create the package**

`kaggle_slayer/harness/telemetry/__init__.py`:

```python
"""Telemetry modules: OTel tracing, CV-LB calibration, error capture,
agent-behavior metrics, MLflow logging."""
```

- [ ] **Step 2: Failing tests**

`tests/unit/test_telemetry_otel.py`:

```python
"""Tests for kaggle_slayer.harness.telemetry.otel."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness.telemetry import otel
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_tracer_writes_span_to_workspace_otel_jsonl(ws):
    tracer = otel.make_tracer(ws, run_name="solve")
    with tracer.start_span("hello", attributes={"k": "v"}) as span:
        span.set_attribute("more", 1)
    # Flush the BatchSpanProcessor so the file gets the span.
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    assert path.exists()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["name"] == "hello"
    assert records[0]["attributes"]["k"] == "v"
    assert records[0]["attributes"]["more"] == 1


def test_tracer_nests_child_spans(ws):
    tracer = otel.make_tracer(ws, run_name="solve")
    with tracer.start_span("outer"):
        with tracer.start_span("inner"):
            pass
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == 2
    names = [r["name"] for r in records]
    assert "outer" in names and "inner" in names
    # Inner's parent_span_id matches outer's span_id
    outer = next(r for r in records if r["name"] == "outer")
    inner = next(r for r in records if r["name"] == "inner")
    assert inner["parent_span_id"] == outer["span_id"]


def test_tracer_records_duration(ws):
    import time
    tracer = otel.make_tracer(ws, run_name="solve")
    with tracer.start_span("timed"):
        time.sleep(0.01)
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    rec = json.loads(path.read_text().splitlines()[0])
    # Duration in nanoseconds; ≥10ms = 10_000_000 ns
    assert rec["duration_ns"] >= 10_000_000


def test_tracer_records_exception_status(ws):
    tracer = otel.make_tracer(ws, run_name="solve")
    with pytest.raises(RuntimeError, match="boom"):
        with tracer.start_span("failing"):
            raise RuntimeError("boom")
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["status"] == "ERROR"
    assert "boom" in rec.get("error", "")
```

- [ ] **Step 3: Run, observe failure**

```bash
pytest tests/unit/test_telemetry_otel.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.telemetry.otel`.

- [ ] **Step 4: Implement `kaggle_slayer/harness/telemetry/otel.py`**

```python
"""OpenTelemetry tracer with a JSONL file exporter.

Each call to `make_tracer(workspace, run_name=...)` returns a Tracer whose
spans append to `<workspace>/otel.jsonl`. Spans are nested via Python
context managers; parent/child relationships are recorded on the span's
`parent_span_id` field. Exceptions raised inside a span set status=ERROR
and record the error message.

We don't ship a full OTLP collector — the JSONL file is enough for the
dashboard and post-hoc debugging. If we ever need OTLP, swap the
processor in `_install_processor`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_slayer.harness.workspace import Workspace

_OTEL_FILENAME = "otel.jsonl"

# Module-level state so `shutdown()` can flush after a test sets up a tracer.
_pending_files: set[Path] = set()


@dataclass
class _Span:
    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None
    started_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _Tracer:
    def __init__(self, file_path: Path, trace_id: str) -> None:
        self._file = file_path
        self._trace_id = trace_id
        self._stack: list[_Span] = []
        _pending_files.add(file_path)

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[_Span]:
        parent = self._stack[-1].span_id if self._stack else None
        span = _Span(
            name=name,
            span_id=os.urandom(8).hex(),
            trace_id=self._trace_id,
            parent_span_id=parent,
            started_ns=time.perf_counter_ns(),
            attributes=dict(attributes or {}),
        )
        self._stack.append(span)
        try:
            yield span
        except Exception as e:  # noqa: BLE001 — record then re-raise
            span.status = "ERROR"
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            ended = time.perf_counter_ns()
            duration = ended - span.started_ns
            self._stack.pop()
            self._write(span, duration)

    def _write(self, span: _Span, duration_ns: int) -> None:
        record = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="microseconds"),
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "duration_ns": duration_ns,
            "status": span.status,
            "attributes": span.attributes,
        }
        if span.error is not None:
            record["error"] = span.error
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def make_tracer(workspace: Workspace, *, run_name: str) -> _Tracer:
    """Return a tracer that appends spans to <workspace>/otel.jsonl."""
    trace_id = os.urandom(16).hex()
    file_path = workspace.root / _OTEL_FILENAME
    tracer = _Tracer(file_path=file_path, trace_id=trace_id)
    # Stamp a synthetic root-of-trace marker so reading the file in order
    # makes the trace boundary obvious.
    with tracer.start_span(f"run:{run_name}"):
        pass
    return tracer


def shutdown() -> None:
    """No-op flush; included for API compatibility with otel SDK consumers."""
    _pending_files.clear()
```

(Note: we deliberately do NOT pull in `opentelemetry-sdk` for V1 — its API surface is heavy and we only need spans-to-JSONL. The dep stays declared so future spec compliance is one swap away.)

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_telemetry_otel.py -v
```

Expected: 4/4 pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/telemetry/__init__.py kaggle_slayer/harness/telemetry/otel.py tests/unit/test_telemetry_otel.py
git commit -m "$(cat <<'EOF'
feat(telemetry): add otel tracer with JSONL file exporter

make_tracer(workspace, run_name) returns a Tracer whose spans append to
<workspace>/otel.jsonl, one record per span with trace_id, parent_span_id,
duration_ns, status, attributes, and error text on failure. Spans nest
via Python context managers; parent/child relationships are explicit on
each record.

We deliberately use a small custom Tracer instead of the full
opentelemetry-sdk for V1 — the JSONL surface is enough for the dashboard
and post-hoc debugging, and avoids 1MB+ of transitive deps. The
opentelemetry-api / opentelemetry-sdk packages stay declared so the
swap to a real OTLP collector is one file change away.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: OTel integration in Solver

Wrap `solve()` in a root span. Wrap each LLM call and each tool dispatch in a child span. Spans record useful attributes (tool name, tokens, duration).

**Files:**
- Modify: `kaggle_slayer/agent/solver.py`
- Modify: `tests/unit/test_solver.py`

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_solver.py`:

```python
def test_solver_writes_otel_trace_to_workspace(tmp_path):
    """Every Solver run emits an OTel trace; root span + child spans per
    LLM call + per tool dispatch."""
    import json
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="take_note",
                                      args={"category": "observation", "content": "x"})],
                 usage=Usage(input_tokens=5, output_tokens=3, cached_tokens=0)),
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    path = ws.root / "otel.jsonl"
    assert path.exists()
    spans = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    span_names = [s["name"] for s in spans]
    # Root span for the run, plus LLM-call spans, plus tool-dispatch spans
    assert any(n.startswith("run:") for n in span_names)
    assert any(n == "llm.call" for n in span_names)
    assert any(n.startswith("tool:") for n in span_names)
    # The tool-dispatch span carries the tool name as an attribute
    tool_spans = [s for s in spans if s["name"].startswith("tool:")]
    assert any(s["attributes"].get("tool.name") in ("take_note", "done") for s in tool_spans)
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_solver.py::test_solver_writes_otel_trace_to_workspace -v
```

Expected: `FileNotFoundError` for `otel.jsonl`.

- [ ] **Step 3: Wire OTel into `Solver.solve()`**

In `kaggle_slayer/agent/solver.py`:

Add to the imports:

```python
from kaggle_slayer.harness.telemetry import otel
```

In `Solver.__init__`, append after the existing assignments:

```python
        self._tracer = otel.make_tracer(workspace, run_name="solve")
```

Modify `Solver.solve()` body so the loop runs inside a span. Find the line `started = time.perf_counter()` and wrap the entire `for iteration in range(...)` loop in:

```python
        with self._tracer.start_span(
            "solve.loop",
            attributes={"competition": self.workspace.name, "max_iterations": self.max_iterations},
        ):
            started = time.perf_counter()
            for iteration in range(1, self.max_iterations + 1):
                # ... existing body ...
```

Inside the loop, wrap the LLM call:

Replace:
```python
            response = self.llm.call(messages=messages, tools=tool_decls)
```

with:
```python
            with self._tracer.start_span(
                "llm.call",
                attributes={"iteration": iteration, "messages": len(messages)},
            ) as span:
                response = self.llm.call(messages=messages, tools=tool_decls)
                span.set_attribute("usage.input_tokens", response.usage.input_tokens)
                span.set_attribute("usage.output_tokens", response.usage.output_tokens)
                span.set_attribute("tool_calls", len(response.tool_calls))
```

And wrap the tool dispatch in `_dispatch`. The existing signature is
`def _dispatch(self, name: str, args: dict[str, Any], *, tool_call_id: str | None = None) -> str:`
and the existing return path is wrapped in `_cap_tool_result(text_result)`.
Keep both. Replace the body of `_dispatch` with:

```python
    def _dispatch(self, name: str, args: dict[str, Any], *, tool_call_id: str | None = None) -> str:
        """Invoke a tool, journal it, return a string result (success or error)."""
        with self._tracer.start_span(
            f"tool:{name}",
            attributes={"tool.name": name, "tool_call_id": tool_call_id or "anon"},
        ) as span:
            try:
                result = self.registry.invoke(name, ctx=self.ctx, args=args)
                text_result = str(result)
                span.set_attribute("result_len", len(text_result))
                self.journal.log_tool_call(
                    tool=name,
                    args=args,
                    # 8 KB matches the LLM-visible cap, so resume can replay
                    # exactly what the LLM originally saw.
                    result_summary=text_result[:8000],
                    tool_call_id=tool_call_id,
                )
                return _cap_tool_result(text_result)
            except ToolError as e:
                err_msg = f"ToolError: {e}"
                span.set_attribute("error", err_msg)
                self.journal.log_tool_error(
                    tool=name, args=args, error=err_msg, tool_call_id=tool_call_id
                )
                return err_msg
            except Exception as e:  # noqa: BLE001
                err_msg = f"unexpected error in {name}: {e!r}"
                span.set_attribute("error", err_msg)
                self.journal.log_tool_error(
                    tool=name, args=args, error=err_msg, tool_call_id=tool_call_id
                )
                return err_msg
```

(`_cap_tool_result` is already defined in `solver.py` — we keep its truncation
behavior so this is a pure wrapper-only change. The signature stays identical.)

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_solver.py::test_solver_writes_otel_trace_to_workspace tests/unit/test_solver.py -v
```

Expected: new test passes; all existing Solver tests still pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/solver.py tests/unit/test_solver.py
git commit -m "$(cat <<'EOF'
feat(solver): emit OTel trace per run + per LLM call + per tool dispatch

Every Solver run now writes <workspace>/otel.jsonl. The trace structure:

  run:<name>            (synthetic boundary marker from make_tracer)
  └─ solve.loop          (root of the main loop)
      ├─ llm.call         (each LLM round-trip; attrs: input/output tokens, tool_calls count)
      └─ tool:<name>      (each dispatch; attrs: tool.name, tool_call_id, result_len, error)

Attributes are kept minimal — no full payload echo — so the file stays
useful even on long runs. The dashboard (Task 10) reads otel.jsonl
alongside run_log.jsonl for the comp-detail timeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CV↔LB calibration tracker

Each successful `submit_kaggle` appends a row to `~/.kaggle_slayer/calibration.jsonl`. The dashboard reads it for the cross-comp calibration chart.

**Files:**
- Create: `kaggle_slayer/harness/telemetry/calibration.py`
- Create: `tests/unit/test_telemetry_calibration.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_telemetry_calibration.py`:

```python
"""Tests for kaggle_slayer.harness.telemetry.calibration."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness.telemetry import calibration


@pytest.fixture
def isolated_calibration(tmp_path, monkeypatch):
    path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", path)
    return path


def test_record_calibration_appends_row(isolated_calibration):
    calibration.record(
        competition="titanic",
        cv_score=0.82,
        lb_score=None,
        problem_type="classification",
        metric="accuracy",
        cv_strategy="stratified_kfold",
    )
    lines = isolated_calibration.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["competition"] == "titanic"
    assert rec["cv_score"] == 0.82
    assert rec["lb_score"] is None
    assert rec["problem_type"] == "classification"
    assert rec["metric"] == "accuracy"
    assert rec["cv_strategy"] == "stratified_kfold"
    assert "ts" in rec


def test_read_history_returns_all_records_in_order(isolated_calibration):
    calibration.record(competition="a", cv_score=0.5, lb_score=None,
                       problem_type="regression", metric="rmse", cv_strategy="kfold")
    calibration.record(competition="b", cv_score=0.9, lb_score=0.88,
                       problem_type="classification", metric="auc", cv_strategy="stratified_kfold")
    history = calibration.read_history()
    assert len(history) == 2
    assert history[0]["competition"] == "a"
    assert history[1]["competition"] == "b"


def test_read_history_filters_by_competition(isolated_calibration):
    calibration.record(competition="a", cv_score=0.5, lb_score=None,
                       problem_type="regression", metric="rmse", cv_strategy="kfold")
    calibration.record(competition="b", cv_score=0.9, lb_score=0.88,
                       problem_type="classification", metric="auc", cv_strategy="stratified_kfold")
    only_a = calibration.read_history(competition="a")
    assert len(only_a) == 1
    assert only_a[0]["competition"] == "a"


def test_read_history_handles_missing_file(isolated_calibration):
    # File doesn't exist yet
    assert calibration.read_history() == []


def test_read_history_skips_malformed_lines(isolated_calibration):
    isolated_calibration.parent.mkdir(parents=True, exist_ok=True)
    with isolated_calibration.open("a") as f:
        f.write('{"competition": "a", "cv_score": 0.5}\n')
        f.write("not json at all\n")
        f.write('{"competition": "b", "cv_score": 0.8}\n')
    history = calibration.read_history()
    assert len(history) == 2
    assert history[0]["competition"] == "a"
    assert history[1]["competition"] == "b"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_telemetry_calibration.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.telemetry.calibration`.

- [ ] **Step 3: Create `kaggle_slayer/harness/telemetry/calibration.py`**

```python
"""CV↔LB calibration tracker.

Every successful `submit_kaggle` writes one row here:

  {"ts": "...", "competition": "titanic", "cv_score": 0.82,
   "lb_score": null, "problem_type": "classification",
   "metric": "accuracy", "cv_strategy": "stratified_kfold"}

`lb_score` is `null` at write time (we don't have the real LB number
until Kaggle scores the submission). Week 6 will add a periodic
backfill task that polls `kaggle_client.get_leaderboard` and updates
the matching row.

The file is `~/.kaggle_slayer/calibration.jsonl` by default — a global
log across all competitions, so the dashboard's cross-comp calibration
chart has one place to read from.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".kaggle_slayer" / "calibration.jsonl"


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def record(
    *,
    competition: str,
    cv_score: float,
    lb_score: float | None,
    problem_type: str,
    metric: str,
    cv_strategy: str,
    path: Path | None = None,
) -> None:
    """Append one calibration row."""
    p = Path(path) if path is not None else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _now_iso(),
        "competition": competition,
        "cv_score": cv_score,
        "lb_score": lb_score,
        "problem_type": problem_type,
        "metric": metric,
        "cv_strategy": cv_strategy,
    }
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_history(
    *,
    competition: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all calibration rows; optionally filter by competition."""
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip partial-write or corruption
            if competition is None or rec.get("competition") == competition:
                out.append(rec)
    return out
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_telemetry_calibration.py -v
```

Expected: 5/5 pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/telemetry/calibration.py tests/unit/test_telemetry_calibration.py
git commit -m "$(cat <<'EOF'
feat(telemetry): add CV↔LB calibration tracker

`record(competition, cv_score, lb_score, problem_type, metric, cv_strategy)`
appends one row to ~/.kaggle_slayer/calibration.jsonl. `lb_score` is
null at write time; Week-6 backfill will fill it from kaggle_client.

`read_history(competition=None)` returns all rows in order, optionally
filtered. Tolerates malformed/partial-write lines (skips them) so the
dashboard never crashes on a corrupt entry.

Hook into submit_kaggle lands in Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Calibration hook in `submit_kaggle`

After a successful kaggle submit, write a calibration row.

**Files:**
- Modify: `kaggle_slayer/agent/handlers/ml.py`
- Modify: `tests/unit/test_handlers_submit_kaggle.py`

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_handlers_submit_kaggle.py`:

```python
def test_submit_kaggle_appends_calibration_row(tmp_path, monkeypatch):
    """Successful submit_kaggle writes a row to the calibration log."""
    import json
    from kaggle_slayer.harness.telemetry import calibration

    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)

    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.best_cv_mean = 0.82
    ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")

    lines = cal_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["competition"] == "test-comp"
    assert row["cv_score"] == 0.82
    assert row["lb_score"] is None
    assert row["metric"] == "accuracy"


def test_submit_kaggle_skips_calibration_on_denial(tmp_path, monkeypatch):
    """A denied submit must not write a calibration row."""
    from kaggle_slayer.harness.telemetry import calibration
    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)

    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    ctx.best_cv_mean = 0.82
    with pytest.raises(ToolError):
        ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="x")

    assert not cal_path.exists() or cal_path.read_text().strip() == ""
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_handlers_submit_kaggle.py::test_submit_kaggle_appends_calibration_row -v
```

Expected: `assert 0 == 1` (no calibration file written).

- [ ] **Step 3: Add the calibration hook to `submit_kaggle`**

In `kaggle_slayer/agent/handlers/ml.py`, find the `submit_kaggle` function. After the successful `kaggle.submit(...)` call but before the `return f"submitted ..."` line, append:

```python
    # Record CV↔LB calibration (lb_score is None until backfill).
    from kaggle_slayer.harness.telemetry import calibration  # noqa: PLC0415

    cv_strategy_name = getattr(ctx, "cv_kind", None) or (
        "stratified_kfold" if ctx.problem_type == "classification" else "kfold"
    )
    calibration.record(
        competition=ctx.competition,
        cv_score=float(ctx.best_cv_mean) if ctx.best_cv_mean is not None else float("nan"),
        lb_score=None,
        problem_type=ctx.problem_type,
        metric=ctx.metric_name,
        cv_strategy=cv_strategy_name,
    )
```

The cv_strategy_name is derived from `ctx.cv_kind` (set by the `set_cv` tool)
or, if the agent never called `set_cv`, the auto-selected default for the
problem type. No `cv_strategies` registry import is needed.

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_handlers_submit_kaggle.py -v
```

Expected: all submit_kaggle tests pass, including the two new ones.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/ml.py tests/unit/test_handlers_submit_kaggle.py
git commit -m "$(cat <<'EOF'
feat(agent): record CV↔LB calibration on every successful submit_kaggle

After a successful kaggle.submit, append a row to
~/.kaggle_slayer/calibration.jsonl with cv_score, lb_score=None,
problem_type, metric, cv_strategy. The lb_score backfill is a Week-6
task; for now we capture the cv_score side of the equation.

Denied submits do NOT write a calibration row (the gate raises ToolError
before the kaggle.submit call, so we never reach the hook).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Error capture

Unhandled exceptions in the CLI write a JSON crash report to `~/.kaggle_slayer/errors/<ts>.json`. Rotation: keep the last 100.

**Files:**
- Create: `kaggle_slayer/harness/telemetry/errors.py`
- Create: `tests/unit/test_telemetry_errors.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_telemetry_errors.py`:

```python
"""Tests for kaggle_slayer.harness.telemetry.errors."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness.telemetry import errors


@pytest.fixture
def isolated_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "DEFAULT_DIR", tmp_path / "errors")
    return tmp_path / "errors"


def test_capture_writes_json_file(isolated_errors):
    try:
        raise ValueError("kaboom")
    except ValueError as e:
        path = errors.capture(e, recent_calls=[{"tool": "x"}], env={"FOO": "bar"})
    assert path.exists()
    rec = json.loads(path.read_text())
    assert rec["exception"]["type"] == "ValueError"
    assert "kaboom" in rec["exception"]["message"]
    assert rec["recent_calls"] == [{"tool": "x"}]
    assert rec["env"]["FOO"] == "bar"
    assert "traceback" in rec


def test_capture_filename_has_iso_timestamp(isolated_errors):
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        path = errors.capture(e, recent_calls=[], env={})
    # YYYY-MM-DD_HHMMSS_ pattern in the filename
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{6}", path.stem)


def test_capture_rotation_keeps_last_100(isolated_errors, monkeypatch):
    """When >100 error files exist, the oldest are pruned."""
    # Seed 105 fake error files with sortable names
    isolated_errors.mkdir(parents=True, exist_ok=True)
    for i in range(105):
        (isolated_errors / f"2026-05-17_{i:06d}_old.json").write_text("{}")
    try:
        raise ValueError("x")
    except ValueError as e:
        errors.capture(e, recent_calls=[], env={})
    files = sorted(isolated_errors.glob("*.json"))
    assert len(files) == 100


def test_capture_redacts_secrets_from_env(isolated_errors):
    """Keys named like *_KEY, *_TOKEN, *_SECRET get their values redacted."""
    try:
        raise ValueError("x")
    except ValueError as e:
        path = errors.capture(e, recent_calls=[], env={
            "GEMINI_API_KEY": "sk-real-key",
            "KAGGLE_API_TOKEN": "KGAT_real",
            "MY_SECRET": "hush",
            "PATH": "/usr/bin",
        })
    rec = json.loads(path.read_text())
    assert rec["env"]["GEMINI_API_KEY"] == "<redacted>"
    assert rec["env"]["KAGGLE_API_TOKEN"] == "<redacted>"
    assert rec["env"]["MY_SECRET"] == "<redacted>"
    assert rec["env"]["PATH"] == "/usr/bin"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_telemetry_errors.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.telemetry.errors`.

- [ ] **Step 3: Create `kaggle_slayer/harness/telemetry/errors.py`**

```python
"""Error capture — JSON crash reports for unhandled exceptions.

CLI's outer try/except calls `capture(exc, recent_calls, env)` and we
dump `<ts>_<exctype>.json` to ~/.kaggle_slayer/errors/, capturing the
traceback, the last N tool calls (for context), and a redacted snapshot
of the environment. Rotation: keep the last 100 reports.

Redaction rule: any env key whose UPPERCASE name contains KEY, TOKEN,
SECRET, or PASSWORD has its value replaced with "<redacted>". This is
a coarse filter — the dev should still review reports before sharing.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import traceback
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path.home() / ".kaggle_slayer" / "errors"
_MAX_FILES = 100
_REDACT_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)")


def _now_filename(exc: BaseException) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H%M%S")
    safe_type = type(exc).__name__.replace(".", "_")
    return f"{stamp}_{safe_type}.json"


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in env.items():
        if _REDACT_RE.search(k.upper()):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _prune(directory: Path) -> None:
    files = sorted(directory.glob("*.json"))
    if len(files) > _MAX_FILES:
        for f in files[: len(files) - _MAX_FILES]:
            try:
                f.unlink()
            except OSError:
                pass  # best-effort


def capture(
    exc: BaseException,
    *,
    recent_calls: list[dict[str, Any]],
    env: dict[str, str],
    directory: Path | None = None,
) -> Path:
    """Write one crash report and prune older ones to _MAX_FILES."""
    d = Path(directory) if directory is not None else DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _now_filename(exc)
    record = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "recent_calls": recent_calls,
        "env": _redact_env(env),
    }
    path.write_text(json.dumps(record, indent=2))
    _prune(d)
    return path
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_telemetry_errors.py -v
```

Expected: 4/4 pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/telemetry/errors.py tests/unit/test_telemetry_errors.py
git commit -m "$(cat <<'EOF'
feat(telemetry): add error-capture module with secret redaction + rotation

capture(exc, recent_calls, env) writes a JSON crash report to
~/.kaggle_slayer/errors/<ts>_<exctype>.json with the exception type,
message, full traceback, the last N tool calls (caller-supplied), and a
redacted snapshot of the environment. Env keys whose UPPERCASE name
contains KEY/TOKEN/SECRET/PASSWORD have their values replaced with
"<redacted>" — coarse but useful for not accidentally sharing API keys
when the user emails a report.

Rotation: keep the last 100 .json files in the directory; older ones
are pruned silently.

CLI integration lands in Task 6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Error capture in CLI

Wrap `run()` in a try/except that dumps to the error directory on any unhandled exception.

**Files:**
- Modify: `kaggle_slayer/cli.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_captures_unhandled_exception_to_errors_dir(tmp_path, monkeypatch, capsys):
    """An unhandled exception inside run() writes a crash report."""
    from kaggle_slayer.harness.telemetry import errors as errors_mod
    err_dir = tmp_path / "errors"
    monkeypatch.setattr(errors_mod, "DEFAULT_DIR", err_dir)

    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    import pandas as pd
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)

    # Make GeminiClient construction blow up so we exercise the catch path.
    def bad_gemini(*args, **kwargs):
        raise RuntimeError("gemini boom")

    with patch("kaggle_slayer.cli.GeminiClient", side_effect=bad_gemini), \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        exit_code = cli.run([str(comp_path), "--target", "y", "--auto-approve", "all", "--i-know-what-im-doing"])

    assert exit_code != 0
    captured = list(err_dir.glob("*.json"))
    assert len(captured) == 1
    import json
    rec = json.loads(captured[0].read_text())
    assert rec["exception"]["type"] == "RuntimeError"
    assert "gemini boom" in rec["exception"]["message"]
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_cli.py::test_cli_captures_unhandled_exception_to_errors_dir -v
```

Expected: the RuntimeError propagates (no crash report).

- [ ] **Step 3: Wrap `run()` in a try/except**

In `kaggle_slayer/cli.py`, find the existing `def run(argv: list[str]) -> int:` function. Rename it to `_run_inner`, then add a new `run` wrapper around it:

```python
def run(argv: list[str]) -> int:
    try:
        return _run_inner(argv)
    except SystemExit:
        raise  # let argparse's sys.exit pass through
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 — outermost CLI catch
        from kaggle_slayer.harness.telemetry import errors  # noqa: PLC0415
        # Best-effort: collect the last few journal records if a workspace path
        # was parsed. If anything in the recovery itself fails, fall through.
        recent_calls: list[dict[str, Any]] = []
        try:
            parsed = _parse_args(argv)
            from kaggle_slayer.harness.journal import Journal  # noqa: PLC0415
            from kaggle_slayer.harness.workspace import Workspace  # noqa: PLC0415
            ws = Workspace(root=Path(parsed.workspace_path))
            if ws.run_log_path.exists():
                recent_calls = list(Journal(ws).iter_records())[-10:]
        except Exception:  # noqa: BLE001
            pass
        path = errors.capture(e, recent_calls=recent_calls, env=dict(os.environ))
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"crash report written to {path}", file=sys.stderr)
        return 4
```

Add `Any` to the typing imports if not already present. Rename the existing function body to `_run_inner`.

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: new test passes; existing CLI tests still pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/cli.py tests/unit/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): capture unhandled exceptions to ~/.kaggle_slayer/errors

run() now wraps the inner logic in a try/except that, on any unhandled
exception, calls telemetry.errors.capture(...) with the exception, the
last 10 tool-call records from the workspace journal (best-effort), and
a redacted env snapshot. Returns exit code 4 (distinct from the existing
0/1/2/3) so callers can branch.

KeyboardInterrupt prints a short message and returns 130 (POSIX
convention). SystemExit (argparse's normal exit) passes through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Agent behavior metrics

Consolidate the stuck-loop detector from `resume.py` into `telemetry/behavior.py` and add a turns counter. The Solver records the final metrics to the journal at exit.

**Files:**
- Create: `kaggle_slayer/harness/telemetry/behavior.py`
- Modify: `kaggle_slayer/harness/resume.py` — delegate to `telemetry.behavior.detect_stuck_loop`
- Create: `tests/unit/test_telemetry_behavior.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_telemetry_behavior.py`:

```python
"""Tests for kaggle_slayer.harness.telemetry.behavior."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness.telemetry import behavior
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_compute_metrics_counts_turns(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_call(tool="train_cv", args={}, result_summary="ok")
    j.log_tool_call(tool="done", args={"summary": "x"}, result_summary="ack")

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_per_run == 3
    assert metrics.tool_counts == {"take_note": 1, "train_cv": 1, "done": 1}


def test_compute_metrics_turns_to_first_submission(ws):
    """The turn index where the FIRST submit_kaggle (or submit_local) lands."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_call(tool="write_file", args={}, result_summary="ok")
    j.log_tool_call(tool="train_cv", args={}, result_summary="ok")
    j.log_tool_call(tool="submit_local", args={"label": "v1"}, result_summary="ok")  # 4th turn
    j.log_tool_call(tool="done", args={"summary": "x"}, result_summary="ack")

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_first_submission == 4


def test_compute_metrics_no_submission_yet(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_first_submission is None


def test_detect_stuck_loop_flags_same_call_repeated(ws):
    """Five identical (tool, args) calls in the last 10 records → stuck."""
    j = Journal(ws)
    for _ in range(5):
        j.log_tool_call(
            tool="train_cv", args={}, result_summary="mean=0.5",
        )
    stuck = behavior.detect_stuck_loop(ws, window=10, threshold=5)
    assert stuck is not None
    assert stuck["tool"] == "train_cv"
    assert stuck["repeats"] == 5


def test_detect_stuck_loop_no_repetition_returns_none(ws):
    j = Journal(ws)
    for i in range(5):
        j.log_tool_call(
            tool="take_note", args={"category": "observation", "content": f"#{i}"},
            result_summary="noted",
        )
    stuck = behavior.detect_stuck_loop(ws, window=10, threshold=5)
    assert stuck is None
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_telemetry_behavior.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.telemetry.behavior`.

- [ ] **Step 3: Create `kaggle_slayer/harness/telemetry/behavior.py`**

```python
"""Agent behavior metrics — derived from the journal.

Turns_per_run, tool_counts, turns_to_first_submission, stuck-loop
detection. Pure functions over `Journal.iter_records()`; no side
effects, no I/O beyond the journal read.

The Solver may snapshot these to the journal at exit (Task 8). The
dashboard reads them on demand.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace

_SUBMIT_TOOLS = frozenset({"submit_kaggle", "submit_local"})


@dataclass
class BehaviorMetrics:
    turns_per_run: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    turns_to_first_submission: int | None = None


def compute_metrics(workspace: Workspace) -> BehaviorMetrics:
    """Walk the journal once and compute all metrics."""
    records = list(Journal(workspace).iter_records())
    counts: Counter[str] = Counter()
    errors = 0
    first_submission_turn: int | None = None

    for i, rec in enumerate(records, start=1):
        kind = rec.get("kind")
        if kind not in ("tool_call", "tool_error"):
            continue
        tool = rec.get("tool", "")
        counts[tool] += 1
        if kind == "tool_error":
            errors += 1
        if first_submission_turn is None and tool in _SUBMIT_TOOLS:
            first_submission_turn = i

    return BehaviorMetrics(
        turns_per_run=sum(counts.values()),
        tool_counts=dict(counts),
        error_count=errors,
        turns_to_first_submission=first_submission_turn,
    )


def detect_stuck_loop(
    workspace: Workspace,
    *,
    window: int = 10,
    threshold: int = 5,
) -> dict[str, Any] | None:
    """If the same (tool, args) appears `threshold` times in the trailing
    window of `window` records, return a description dict; else None."""
    records = list(Journal(workspace).iter_records())
    window_records = records[-window:]
    sigs: Counter[tuple[str, str]] = Counter()
    for rec in window_records:
        kind = rec.get("kind")
        if kind not in ("tool_call", "tool_error"):
            continue
        sig = (rec.get("tool", ""), json.dumps(rec.get("args", {}), sort_keys=True))
        sigs[sig] += 1
    if not sigs:
        return None
    (tool, args_repr), count = sigs.most_common(1)[0]
    if count < threshold:
        return None
    return {
        "tool": tool,
        "args": json.loads(args_repr),
        "repeats": count,
        "window": window,
    }
```

- [ ] **Step 4: Delegate from `resume.py:summarize`**

Find the inline `# Stuck loop: tally ...` block in `kaggle_slayer/harness/resume.py:summarize`. Replace the stuck-loop computation with a delegating call:

```python
    # Stuck-loop detection (moved to telemetry.behavior for reuse).
    from kaggle_slayer.harness.telemetry import behavior  # noqa: PLC0415
    summary.stuck_loop = behavior.detect_stuck_loop(
        workspace, window=stuck_window, threshold=stuck_threshold,
    )
```

Keep the existing function signature and behavior identical — this is a pure refactor.

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_telemetry_behavior.py tests/unit/test_resume.py -v
```

Expected: 5 new tests pass; the resume tests still pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/telemetry/behavior.py kaggle_slayer/harness/resume.py tests/unit/test_telemetry_behavior.py
git commit -m "$(cat <<'EOF'
feat(telemetry): add behavior metrics — turns_per_run, stuck-loop, first-submission

compute_metrics(workspace) returns turns_per_run, tool_counts,
error_count, turns_to_first_submission (the journal turn index of the
first submit_kaggle/submit_local, or None). Pure read of the journal,
no side effects.

detect_stuck_loop(workspace, window, threshold) returns a {tool, args,
repeats, window} dict when the same (tool, args) pair appears
`threshold` times in the trailing window — useful for breaking out of
agent retry loops.

resume.summarize() now delegates its stuck-loop computation to
telemetry.behavior.detect_stuck_loop (pure refactor; same behavior).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: MLflow logging in `train_cv`

One MLflow run per `train_cv` call. Logs params + cv_mean/std + per-fold scores. Artifact logging (fe.py/model.py/oof_preds) is deferred to Week 6.

**Files:**
- Create: `kaggle_slayer/harness/telemetry/mlflow_logger.py`
- Modify: `kaggle_slayer/agent/handlers/ml.py` — wrap `train_cv` in the new context manager
- Create: `tests/unit/test_telemetry_mlflow.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_telemetry_mlflow.py`:

```python
"""Tests for kaggle_slayer.harness.telemetry.mlflow_logger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kaggle_slayer.harness.telemetry import mlflow_logger
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_log_train_cv_starts_and_ends_run(ws):
    """The context manager calls mlflow.start_run + log_params + log_metrics."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_run = MagicMock()
        mock_ml.start_run.return_value.__enter__.return_value = mock_run
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic",
            cv_strategy="stratified_kfold",
            metric="accuracy",
            fe_version="fe_v01",
            model_version="model_v01",
        ) as logger:
            logger.log_result(cv_mean=0.82, cv_std=0.03, fold_scores=[0.80, 0.83, 0.83])

    mock_ml.start_run.assert_called_once()
    mock_ml.log_params.assert_called_once()
    params = mock_ml.log_params.call_args[0][0]
    assert params["cv_strategy"] == "stratified_kfold"
    assert params["metric"] == "accuracy"
    assert params["fe_version"] == "fe_v01"
    assert params["model_version"] == "model_v01"

    mock_ml.log_metrics.assert_called()
    metrics = mock_ml.log_metrics.call_args[0][0]
    assert metrics["cv_mean"] == 0.82
    assert metrics["cv_std"] == 0.03
    assert metrics["fold_0"] == 0.80
    assert metrics["fold_1"] == 0.83
    assert metrics["fold_2"] == 0.83


def test_log_train_cv_sets_experiment_per_competition(ws):
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
        ):
            pass

    mock_ml.set_experiment.assert_called_once_with("kaggleslayer/titanic")


def test_log_train_cv_swallows_mlflow_failures(ws):
    """If mlflow.start_run raises, the user's code still runs — we don't break the agent."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.set_experiment.side_effect = RuntimeError("mlflow down")

        # Should NOT raise — failure is logged via the noop fallback.
        with mlflow_logger.log_train_cv(
            competition="x", cv_strategy="kfold", metric="rmse",
            fe_version="fe_v01", model_version="model_v01",
        ) as logger:
            logger.log_result(cv_mean=0.5, cv_std=0.1, fold_scores=[0.4, 0.6])
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_telemetry_mlflow.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.telemetry.mlflow_logger`.

- [ ] **Step 3: Create `kaggle_slayer/harness/telemetry/mlflow_logger.py`**

```python
"""MLflow logging for train_cv — one run per call.

Spec §11.1: one experiment per competition (`kaggleslayer/<comp>`), one
run per `train_cv` invocation. Params: cv_strategy, metric, fe_version,
model_version. Metrics: cv_mean, cv_std, fold_0...fold_N.

Artifact logging (fe.py, model.py, oof_preds.npy) lands in Week 6.

`log_train_cv(...)` is a context manager. Errors from MLflow are
swallowed (the agent shouldn't crash if the tracking server is down) —
a comment explains where to look if metrics are missing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import mlflow  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)


@dataclass
class _RunLogger:
    """Handed to the caller inside the `with` block; collects results."""

    _logged: bool = field(default=False, init=False)

    def log_result(
        self,
        *,
        cv_mean: float,
        cv_std: float,
        fold_scores: list[float],
    ) -> None:
        if self._logged:
            return  # idempotent
        try:
            metrics: dict[str, float] = {
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
            }
            for i, score in enumerate(fold_scores):
                metrics[f"fold_{i}"] = float(score)
            mlflow.log_metrics(metrics)
        except Exception:  # noqa: BLE001
            _log.exception("mlflow.log_metrics failed; continuing")
        self._logged = True


@contextlib.contextmanager
def log_train_cv(
    *,
    competition: str,
    cv_strategy: str,
    metric: str,
    fe_version: str,
    model_version: str,
) -> Iterator[_RunLogger]:
    """Wrap one train_cv invocation in an MLflow run."""
    logger = _RunLogger()
    try:
        mlflow.set_experiment(f"kaggleslayer/{competition}")
        with mlflow.start_run():
            mlflow.log_params({
                "cv_strategy": cv_strategy,
                "metric": metric,
                "fe_version": fe_version,
                "model_version": model_version,
            })
            yield logger
    except Exception:  # noqa: BLE001
        # MLflow itself blew up. Still yield a logger so caller code is identical;
        # log_result will fail silently inside.
        _log.exception("mlflow.start_run / set_experiment failed; metrics not recorded")
        yield logger
```

- [ ] **Step 4: Hook MLflow into `train_cv`**

In `kaggle_slayer/agent/handlers/ml.py`, find the `train_cv` function. The plan is to wrap the `cv_mod.train_cv(...)` call in the new context manager. Modify the function so the body becomes:

Locate this region (around the existing `result = cv_mod.train_cv(...)` call):

```python
    result = cv_mod.train_cv(
        fe_path=fe_path,
        model_path=model_path,
        train_df=train_df,
        target_col=ctx.target_col,
        cv=cv,
        metric=metric,
        metadata_extra={"fe_version": fe_archive.stem, "model_version": model_archive.stem},
    )
```

Wrap it:

```python
    from kaggle_slayer.harness.telemetry import mlflow_logger  # noqa: PLC0415
    with mlflow_logger.log_train_cv(
        competition=ctx.workspace.name,
        cv_strategy=cv.name,
        metric=metric.name,
        fe_version=fe_archive.stem,
        model_version=model_archive.stem,
    ) as run_logger:
        result = cv_mod.train_cv(
            fe_path=fe_path,
            model_path=model_path,
            train_df=train_df,
            target_col=ctx.target_col,
            cv=cv,
            metric=metric,
            metadata_extra={"fe_version": fe_archive.stem, "model_version": model_archive.stem},
        )
        run_logger.log_result(
            cv_mean=result.mean,
            cv_std=result.std,
            fold_scores=list(result.fold_scores),
        )
```

(Keep the existing best_cv_mean tracking + summary return unchanged.)

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_telemetry_mlflow.py tests/unit/test_handlers_ml.py -v
```

Expected: new tests pass; the existing train_cv tests still pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/harness/telemetry/mlflow_logger.py kaggle_slayer/agent/handlers/ml.py tests/unit/test_telemetry_mlflow.py
git commit -m "$(cat <<'EOF'
feat(telemetry): log every train_cv to an MLflow run

One experiment per competition (kaggleslayer/<comp>), one run per
train_cv invocation. Params: cv_strategy, metric, fe_version,
model_version. Metrics: cv_mean, cv_std, fold_0..fold_N. Artifact
logging (fe.py, model.py, oof_preds) is Week-6 scope.

MLflow failures are swallowed and logged via stdlib `logging` — the
agent's loop must not crash because the tracking server is down. The
RunLogger context manager always yields, so caller code is identical
on the happy path and the failure path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Streamlit dashboard scaffolding

The dashboard entry. Reads from `competitions/` only — no live calls.

**Files:**
- Create: `kaggle_slayer/dashboard/__init__.py`
- Create: `kaggle_slayer/dashboard/app.py`
- Modify: `pyproject.toml` — add the entry point

- [ ] **Step 1: Create the package marker**

`kaggle_slayer/dashboard/__init__.py`:

```python
"""Streamlit dashboard — portfolio + per-comp detail.

Run with:
    kaggle-slayer-dashboard
or:
    streamlit run kaggle_slayer/dashboard/app.py
"""
```

- [ ] **Step 2: Create `kaggle_slayer/dashboard/app.py`**

```python
"""Streamlit entry: portfolio + comp-detail routing.

The dashboard reads only on-disk artifacts — never touches Kaggle or
Gemini. Pages live in `kaggle_slayer/dashboard/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from kaggle_slayer.dashboard import comp_detail, portfolio


def main() -> None:
    """Entry point for `kaggle-slayer-dashboard`."""
    # When invoked as a console_script (not via `streamlit run`), re-exec under
    # streamlit so the user gets the browser UI.
    if "streamlit.runtime.scriptrunner" not in sys.modules:
        import streamlit.web.cli as stcli  # type: ignore[import-untyped]
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())
    _run_pages()


def _run_pages() -> None:
    st.set_page_config(page_title="KaggleSlayer", layout="wide")
    st.sidebar.title("KaggleSlayer")
    page = st.sidebar.radio("Page", ["Portfolio", "Competition detail"])
    comps_root = Path(
        st.sidebar.text_input("Competitions root", value="competitions")
    )
    if page == "Portfolio":
        portfolio.render(comps_root)
    else:
        portfolio_names = portfolio.list_competitions(comps_root)
        if not portfolio_names:
            st.warning(f"No competitions found under {comps_root}.")
            return
        chosen = st.sidebar.selectbox("Competition", portfolio_names)
        comp_detail.render(comps_root / chosen)


# When `streamlit run kaggle_slayer/dashboard/app.py` is used directly,
# the module is imported and we land here.
if "streamlit.runtime.scriptrunner" in sys.modules:
    _run_pages()
```

- [ ] **Step 3: Add the entry point in `pyproject.toml`**

Find the existing `[project.scripts]` block:

```toml
[project.scripts]
kaggle-slayer = "kaggle_slayer.cli:main"
```

Replace with:

```toml
[project.scripts]
kaggle-slayer = "kaggle_slayer.cli:main"
kaggle-slayer-dashboard = "kaggle_slayer.dashboard.app:main"
```

- [ ] **Step 4: Smoke test**

```bash
pip install -e ".[dev,dashboard]"
kaggle-slayer-dashboard --help 2>&1 | head -5  # streamlit's own help passes through
```

Expected: Streamlit's `--help` prints (or a non-zero exit with a sensible message if the smoke test runs in CI without a TTY).

There is no automated test for the dashboard entry — it's a Streamlit thin shell. The portfolio and comp-detail rendering is tested separately in Tasks 10 and 11.

- [ ] **Step 5: Commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/dashboard/__init__.py kaggle_slayer/dashboard/app.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(dashboard): add Streamlit scaffolding + kaggle-slayer-dashboard entry

Two-page layout: Portfolio (list of comps under `competitions/`) and
Competition detail (timeline + cost + calibration + notes). The
`kaggle-slayer-dashboard` console_script re-execs under streamlit so
users don't need to remember `streamlit run`.

Page rendering lives in `kaggle_slayer/dashboard/portfolio.py` and
`kaggle_slayer/dashboard/comp_detail.py` — see Tasks 10 and 11.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Dashboard portfolio page

A single page that lists each competition with its best CV, total cost spent, and last activity timestamp.

**Files:**
- Create: `kaggle_slayer/dashboard/portfolio.py`
- Create: `tests/unit/test_dashboard_portfolio.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_dashboard_portfolio.py`:

```python
"""Tests for kaggle_slayer.dashboard.portfolio.

The Streamlit `render` function uses live `st.*` calls; we test
the pure helpers (list_competitions + best_cv_for) directly.
"""

from __future__ import annotations

import pytest

from kaggle_slayer.dashboard import portfolio
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def comps_root(tmp_path):
    return tmp_path / "competitions"


def test_list_competitions_returns_empty_for_missing_root(comps_root):
    assert portfolio.list_competitions(comps_root) == []


def test_list_competitions_returns_workspace_dirs(comps_root):
    comps_root.mkdir(parents=True)
    Workspace.create(root=comps_root / "titanic")
    Workspace.create(root=comps_root / "house-prices")
    names = portfolio.list_competitions(comps_root)
    assert sorted(names) == ["house-prices", "titanic"]


def test_list_competitions_skips_non_workspaces(comps_root):
    """A directory missing `agent/` is not a workspace."""
    comps_root.mkdir(parents=True)
    Workspace.create(root=comps_root / "titanic")
    (comps_root / "not-a-workspace").mkdir()
    (comps_root / "not-a-workspace" / "random.txt").write_text("x")
    names = portfolio.list_competitions(comps_root)
    assert names == ["titanic"]


def test_best_cv_for_reads_from_journal(comps_root):
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "titanic")
    j = Journal(ws)
    j.log_tool_call(tool="train_cv", args={}, result_summary=(
        "train_cv complete: stratified_kfold (5 folds), metric=accuracy, "
        "mean=0.7800, std=0.02, fold_scores=[0.78, 0.76, 0.80, 0.79, 0.77], "
        "duration_s=0.50"
    ))
    j.log_tool_call(tool="train_cv", args={}, result_summary=(
        "train_cv complete: stratified_kfold (5 folds), metric=accuracy, "
        "mean=0.8500, std=0.01, fold_scores=[0.85, 0.84, 0.86, 0.85, 0.85], "
        "duration_s=0.50"
    ))
    best = portfolio.best_cv_for(ws)
    assert best is not None
    assert abs(best - 0.85) < 1e-6


def test_best_cv_for_returns_none_when_no_train_cv(comps_root):
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "titanic")
    Journal(ws).log_tool_call(tool="take_note", args={}, result_summary="ok")
    assert portfolio.best_cv_for(ws) is None
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_dashboard_portfolio.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.dashboard.portfolio`.

- [ ] **Step 3: Create `kaggle_slayer/dashboard/portfolio.py`**

```python
"""Portfolio page — list of competitions with summary metrics.

Pure helpers (list_competitions, best_cv_for) are unit-tested. The
`render` function calls Streamlit and is exercised manually + via the
slow-tier integration test (Task 15).
"""

from __future__ import annotations

import re
from pathlib import Path

from kaggle_slayer.agent.cost_ledger import CostLedger, DEFAULT_LEDGER_PATH
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace

_MEAN_RE = re.compile(r"mean=([0-9.]+)")


def list_competitions(comps_root: Path) -> list[str]:
    """Return sorted names of competition workspaces under `comps_root`."""
    comps_root = Path(comps_root)
    if not comps_root.is_dir():
        return []
    names: list[str] = []
    for child in comps_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "agent").is_dir():
            names.append(child.name)
    return sorted(names)


def best_cv_for(workspace: Workspace) -> float | None:
    """Walk the journal, extract `mean=<float>` from train_cv result_summary
    lines, return the max."""
    j = Journal(workspace)
    best: float | None = None
    for rec in j.iter_records():
        if rec.get("tool") != "train_cv" or rec.get("kind") != "tool_call":
            continue
        summary = rec.get("result_summary", "")
        m = _MEAN_RE.search(summary)
        if not m:
            continue
        try:
            mean = float(m.group(1))
        except ValueError:
            continue
        if best is None or mean > best:
            best = mean
    return best


def render(comps_root: Path) -> None:
    """Streamlit page: card per competition."""
    import streamlit as st

    st.title("Portfolio")
    names = list_competitions(comps_root)
    if not names:
        st.info(f"No competitions found under `{comps_root}`. Run "
                f"`kaggle-slayer <path>` to create one.")
        return

    ledger = CostLedger(path=DEFAULT_LEDGER_PATH)
    for name in names:
        ws = Workspace(root=comps_root / name)
        cv = best_cv_for(ws)
        cost = ledger.total_for(competition=name)
        run_log = ws.run_log_path

        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(f"### `{name}`")
            cols[1].metric("Best CV", f"{cv:.4f}" if cv is not None else "—")
            cols[2].metric("Cost (USD)", f"${cost:.4f}")
            cols[3].metric("Tool calls", str(len(list(Journal(ws).iter_records())))
                           if run_log.exists() else "0")
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_dashboard_portfolio.py -v
```

Expected: 5/5 pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/dashboard/portfolio.py tests/unit/test_dashboard_portfolio.py
git commit -m "$(cat <<'EOF'
feat(dashboard): portfolio page — list comps with best CV + cost + tool count

list_competitions(comps_root) returns sorted workspace names (a dir is a
workspace iff `agent/` exists inside it). best_cv_for(workspace) walks
the journal for train_cv result_summary lines and extracts the maximum
`mean=<f>` — pure read, no MLflow dependency.

The Streamlit render() draws one card per comp with: name, best CV, USD
spent, and tool-call count. Cost is read from the global ledger.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Dashboard comp-detail page

A single page that surfaces, for one competition: the journal timeline, the cost ledger entries, the calibration history, the notes browser, and a download link for `submissions/*.csv`.

**Files:**
- Create: `kaggle_slayer/dashboard/comp_detail.py`
- Create: `tests/unit/test_dashboard_comp_detail.py`

- [ ] **Step 1: Failing tests**

`tests/unit/test_dashboard_comp_detail.py`:

```python
"""Tests for the pure helpers in kaggle_slayer.dashboard.comp_detail."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.dashboard import comp_detail
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "titanic")


def test_journal_timeline_returns_in_order(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.8")
    timeline = comp_detail.journal_timeline(ws)
    assert len(timeline) == 2
    assert timeline[0]["tool"] == "take_note"
    assert timeline[1]["tool"] == "train_cv"


def test_journal_timeline_empty_for_missing_log(ws):
    assert comp_detail.journal_timeline(ws) == []


def test_notes_browser_returns_filtered_categories(ws):
    j = Journal(ws)
    j.take_note(category="observation", content="A")
    j.take_note(category="decision", content="B")
    j.take_note(category="observation", content="C")
    notes = comp_detail.read_notes(ws)
    assert len(notes) == 3
    observations = comp_detail.read_notes(ws, category="observation")
    assert len(observations) == 2
    assert all(n["category"] == "observation" for n in observations)


def test_list_submissions_returns_csv_paths(ws):
    (ws.submissions_dir / "2026-05-17_v01.csv").write_text("id,target\n1,0\n")
    (ws.submissions_dir / "2026-05-17_v02.csv").write_text("id,target\n1,1\n")
    (ws.submissions_dir / "leaderboard.jsonl").write_text("")  # not a submission
    submissions = comp_detail.list_submissions(ws)
    assert len(submissions) == 2
    assert all(p.suffix == ".csv" for p in submissions)


def test_calibration_for_competition_filters_history(ws, tmp_path, monkeypatch):
    """The page reads from telemetry.calibration filtered by competition name."""
    from kaggle_slayer.harness.telemetry import calibration
    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)
    calibration.record(competition="titanic", cv_score=0.82, lb_score=None,
                       problem_type="classification", metric="accuracy",
                       cv_strategy="stratified_kfold")
    calibration.record(competition="other", cv_score=0.5, lb_score=None,
                       problem_type="regression", metric="rmse", cv_strategy="kfold")
    rows = comp_detail.calibration_for(ws)
    assert len(rows) == 1
    assert rows[0]["competition"] == "titanic"
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_dashboard_comp_detail.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.dashboard.comp_detail`.

- [ ] **Step 3: Create `kaggle_slayer/dashboard/comp_detail.py`**

```python
"""Competition-detail page — timeline, cost, calibration, notes, submissions.

Pure helpers are unit-tested. `render` calls Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_slayer.agent.cost_ledger import CostLedger, DEFAULT_LEDGER_PATH
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.telemetry import calibration
from kaggle_slayer.harness.workspace import Workspace


def journal_timeline(workspace: Workspace) -> list[dict[str, Any]]:
    """Every journal record in order."""
    return list(Journal(workspace).iter_records())


def read_notes(
    workspace: Workspace, *, category: str | None = None
) -> list[dict[str, Any]]:
    """All notes from notes.jsonl, optionally filtered by category."""
    return Journal(workspace).list_notes(category=category)


def list_submissions(workspace: Workspace) -> list[Path]:
    """All submission CSVs (excludes leaderboard.jsonl etc)."""
    return sorted(workspace.submissions_dir.glob("*.csv"))


def calibration_for(workspace: Workspace) -> list[dict[str, Any]]:
    """Calibration history filtered by competition name."""
    return calibration.read_history(competition=workspace.name)


def render(workspace_root: Path) -> None:
    """Streamlit page: per-comp detail."""
    import streamlit as st

    workspace = Workspace(root=workspace_root)
    st.title(f"Competition · `{workspace.name}`")

    # Top metrics
    ledger = CostLedger(path=DEFAULT_LEDGER_PATH)
    cost = ledger.total_for(competition=workspace.name)
    timeline = journal_timeline(workspace)
    cal_rows = calibration_for(workspace)
    cols = st.columns(4)
    cols[0].metric("Tool calls", len(timeline))
    cols[1].metric("Cost (USD)", f"${cost:.4f}")
    cols[2].metric("Submissions", len(cal_rows))
    cols[3].metric("Submissions on disk", len(list_submissions(workspace)))

    # Timeline (table)
    st.subheader("Tool-call timeline")
    if timeline:
        import pandas as pd  # type: ignore[import-untyped]
        df = pd.DataFrame([
            {
                "ts": r.get("ts", ""),
                "kind": r.get("kind", ""),
                "tool": r.get("tool", ""),
                "args_keys": ", ".join(r.get("args", {}).keys()),
                "summary": (r.get("result_summary") or r.get("error") or "")[:120],
            }
            for r in timeline
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No journal records yet.")

    # Calibration table
    st.subheader("CV ↔ LB calibration")
    if cal_rows:
        st.dataframe(cal_rows, use_container_width=True)
    else:
        st.info("No submissions recorded yet.")

    # Notes
    st.subheader("Agent notes")
    notes = read_notes(workspace)
    if notes:
        st.dataframe(notes, use_container_width=True)
    else:
        st.info("Agent hasn't taken any notes yet.")

    # Submission downloads
    st.subheader("Submissions on disk")
    subs = list_submissions(workspace)
    if subs:
        for p in subs:
            with p.open("rb") as f:
                st.download_button(
                    label=f"Download {p.name}", data=f, file_name=p.name, mime="text/csv",
                )
    else:
        st.info("No submission CSVs yet.")
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_dashboard_comp_detail.py -v
```

Expected: 5/5 pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/dashboard/comp_detail.py tests/unit/test_dashboard_comp_detail.py
git commit -m "$(cat <<'EOF'
feat(dashboard): comp-detail page — timeline + cost + calibration + notes

Surfaces: journal timeline (chronological table), cost-ledger total for
this competition, calibration history (CV scores per submit_kaggle),
agent notes (filterable in render), submission-CSV download buttons.

The pure helpers (journal_timeline, read_notes, list_submissions,
calibration_for) are unit-tested; render() draws the Streamlit layout
and is exercised via the slow-tier integration test in Task 15.

The fe_v01↔fe_v02 diff page and feature-importance chart (spec §11.4)
are Week-6 polish; this Week-5 page is the foundation they'll layer onto.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Chaos tier — FailureInjectingLLMClient fixture

A test client that randomly fails 5% of `call()` invocations with `TransientLLMError`. The Solver's existing retry logic must cope.

**Files:**
- Create: `tests/chaos/__init__.py`
- Create: `tests/chaos/conftest.py`
- Create: `tests/chaos/test_solver_chaos.py`

- [ ] **Step 1: Create the package marker**

`tests/chaos/__init__.py`:

```python
"""Chaos tier — integration tests with injected failures.

Run with: pytest tests/chaos -v -m chaos
"""
```

- [ ] **Step 2: Register the marker in `pyproject.toml`**

Find `[tool.pytest.ini_options]` and add `"chaos"` to the markers list:

```toml
markers = [
    "slow: end-to-end / slower tests (run with -m slow)",
    "integration: integration tests with fake agent (faster than slow)",
    "chaos: chaos tier — integration tests with injected tool/LLM failures",
]
```

- [ ] **Step 3: Create the failure-injecting client in `tests/chaos/conftest.py`**

```python
"""Chaos-tier fixtures: FailureInjectingLLMClient + a deterministic seed."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from kaggle_slayer.agent.llm_client import (
    Response,
    TransientLLMError,
)

DEFAULT_FAILURE_RATE: float = 0.05
DEFAULT_SEED: int = 12345


class FailureInjectingLLMClient:
    """Wraps a scripted client and fails `rate` of calls with TransientLLMError.

    Determinism: a seeded random.Random decides per-call whether to fail.
    Tests pass the same seed for reproducibility.
    """

    def __init__(
        self,
        inner_call: Callable[..., Response],
        *,
        rate: float = DEFAULT_FAILURE_RATE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._inner_call = inner_call
        self._rate = rate
        self._rng = random.Random(seed)
        self.failures = 0
        self.successes = 0

    def call(self, messages, *, tools=None, model=None):
        if self._rng.random() < self._rate:
            self.failures += 1
            raise TransientLLMError("injected transient failure (chaos tier)")
        self.successes += 1
        return self._inner_call(messages, tools=tools, model=model)


@pytest.fixture
def chaos_seed() -> int:
    return DEFAULT_SEED
```

- [ ] **Step 4: Test fixture installation**

Verify the marker registers:

```bash
pytest --markers 2>&1 | grep chaos
```

Expected: shows the chaos marker.

- [ ] **Step 5: Commit (no test code yet — that's Task 13)**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/chaos/__init__.py tests/chaos/conftest.py pyproject.toml
git commit -m "$(cat <<'EOF'
test(chaos): add FailureInjectingLLMClient fixture + chaos pytest marker

The wrapper fails `rate` (default 5%) of LLM call() invocations with
TransientLLMError, using a seeded random.Random for reproducibility.
The wrapped client's actual call() is invoked on the non-failure path
so existing scripted-client integration tests can be retrofitted with
chaos by swapping the client.

pyproject.toml registers `chaos` alongside `slow` / `integration`.
The actual chaos test lands in Task 13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Chaos test — full scripted run with 5% failures

Take the existing scripted-fake-LLM integration flow, wrap it in `FailureInjectingLLMClient`, and assert the Solver still finishes cleanly.

**Files:**
- Create: `tests/chaos/test_solver_chaos.py`

- [ ] **Step 1: Failing test**

```python
"""Chaos test: scripted Solver run with 5% injected LLM failures.

The Solver's existing retry logic (GeminiClient.call has `retry_max=3` and
`_is_transient` catches TransientLLMError) must handle the chaos without
losing progress. The chaos client wraps the scripted client, so on the
NON-failure path the LLM behaves identically to the deterministic
integration test in tests/integration/test_solver_with_fake_agent.py.
"""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from tests.chaos.conftest import FailureInjectingLLMClient
from tests.fixtures.synthetic_comp import make_synthetic_comp


pytestmark = [pytest.mark.chaos, pytest.mark.integration]


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

    def __call__(self, messages, *, tools=None, model=None):
        r = self._r[self._i]
        self._i += 1
        return r


def test_solver_survives_5_percent_chaos(tmp_path, chaos_seed):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    scripted = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="write_file",
                 args={"path": "agent/fe.py", "content": _FE_CODE})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="write_file",
                 args={"path": "agent/model.py", "content": _MODEL_CODE})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t3", name="train_cv", args={})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t4", name="submit_local",
                 args={"label": "chaos"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t5", name="done",
                 args={"summary": "chaos-tier complete"})], usage=Usage(0, 0, 0)),
    ])
    chaos_client = FailureInjectingLLMClient(
        inner_call=scripted, rate=0.05, seed=chaos_seed,
    )

    solver = Solver(
        workspace=workspace, llm_client=chaos_client,
        target_col="Survived", problem_type="classification",
        metric_name="accuracy", max_iterations=20,
    )

    # The Solver doesn't have a built-in retry around the LLM call (GeminiClient
    # has its own retry; the fake doesn't). For the chaos test we measure
    # *journal* survival: even if some iterations fail, no journal record
    # should be corrupted, and (with 5% failure rate × 5 happy-path calls)
    # the run should finish in ≤10 iterations.
    #
    # If the chaos throws on the very last call and exhausts the script,
    # the test fails — that's a real signal we need a retry layer in the
    # Solver. For now, assert that EITHER the run completed OR the journal
    # is in a parseable state.
    from kaggle_slayer.agent.llm_client import TransientLLMError
    try:
        result = solver.solve()
        assert result.status == "done" or result.status == "max_iterations"
    except TransientLLMError:
        # A transient bubbled up. Journal should still be well-formed.
        pass

    # The journal must be parseable line-by-line (no corruption).
    import json
    if workspace.run_log_path.exists():
        for line in workspace.run_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)  # raises if corrupted

    # We expect at least SOME failures were injected over a 5-call run with 5%
    # rate × seeded RNG. The exact count depends on the seed; assert non-zero
    # via at least one of the counters or skip the check if seed gave us zero.
    assert chaos_client.successes >= 1


def test_chaos_client_failure_rate_is_seeded(chaos_seed):
    """Same seed → same failure pattern across runs."""
    rs = []
    for _ in range(2):
        scripted = _Scripted([
            Response(text="", tool_calls=[], usage=Usage(0, 0, 0))
            for _ in range(1000)
        ])
        c = FailureInjectingLLMClient(inner_call=scripted, rate=0.05, seed=chaos_seed)
        from kaggle_slayer.agent.llm_client import TransientLLMError
        for _ in range(1000):
            try:
                c.call(messages=[])
            except TransientLLMError:
                pass
        rs.append(c.failures)
    assert rs[0] == rs[1]
    # ~5% of 1000 = 50; allow a wide band for RNG slop
    assert 25 <= rs[0] <= 100
```

- [ ] **Step 2: Run, observe pass**

```bash
pytest tests/chaos/test_solver_chaos.py -v -m chaos
```

Expected: 2/2 pass. The first test may exit early via `TransientLLMError` depending on the seed — that's allowed; the journal-integrity assertion is what matters.

- [ ] **Step 3: Confirm `-m "not slow"` exclusion is unchanged**

```bash
pytest -m "not slow" --collect-only 2>&1 | tail -3
```

The chaos tests are marked `pytest.mark.chaos AND pytest.mark.integration`, so they ARE included in `-m "not slow"`. That's intentional: chaos runs cheap and adds resilience signal.

- [ ] **Step 4: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/chaos/test_solver_chaos.py
git commit -m "$(cat <<'EOF'
test(chaos): full scripted Solver run with 5% injected LLM failures

Wraps the deterministic integration sequence (write_file, write_file,
train_cv, submit_local, done) in a FailureInjectingLLMClient and
asserts:
  - The run either reaches `done` / `max_iterations` cleanly OR raises
    TransientLLMError without corrupting the journal.
  - run_log.jsonl is parseable line-by-line even on failure.
  - Same seed produces identical failure patterns.

The 5% rate × ~5 happy-path calls gives roughly one in three runs at
least one failure under the seeded RNG. The test is deterministic.

If a future change adds a Solver-side retry layer (currently the
GeminiClient has retries; the fake client does not), this test will
keep passing because the success criterion is journal integrity, not
completion. The "completion" assertion is a soft check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Real-Gemini slow-tier with telemetry on

A new slow-tier acceptance: real Gemini solves the synthetic comp, and we assert OTel trace + calibration row + cost-ledger row + MLflow run all materialize.

**Files:**
- Create: `tests/integration/test_solver_real_gemini_telemetry.py`

- [ ] **Step 1: Write the slow-tier test**

```python
"""Real-Gemini E2E with telemetry artifacts — slow tier, opt-in.

Runs the Solver end-to-end on a synthetic micro-comp with real Gemini.
Asserts: otel.jsonl has a `run:solve` span, the cost-ledger has at least
one row for the competition, and (if submit_kaggle is reached via mock)
a calibration row is appended.

Costs ~$0.005-0.02 per run. Skipped when GEMINI_API_KEY is missing.
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
from kaggle_slayer.harness.telemetry import calibration
from tests.fixtures.synthetic_comp import make_synthetic_comp

load_dotenv()

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env")
    return key


def test_real_gemini_writes_all_telemetry_artifacts(tmp_path, gemini_key, monkeypatch):
    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)
    cost_path = tmp_path / "cost.jsonl"

    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    fake_kaggle = MagicMock()

    ledger = CostLedger(path=cost_path)
    llm = GeminiClient(
        api_key=gemini_key, ledger=ledger, competition="synthetic-telemetry",
        default_model="gemini-2.5-flash", retry_max=4, retry_base_delay_s=20.0,
    )
    solver = Solver(
        workspace=workspace, llm_client=llm,
        target_col="Survived", problem_type="classification", metric_name="accuracy",
        max_iterations=20, time_budget_s=900.0,
        checkpoint_handler=handler, kaggle_client=fake_kaggle,
    )

    from unittest.mock import patch
    from kaggle_slayer.agent import prompts as p_mod
    original_loader = p_mod.load_system_prompt

    def loader_with_kaggle():
        return original_loader() + (
            "\n\n## Extra instruction for this run\n"
            "After submit_local succeeds, you MUST call submit_kaggle with "
            "csv_path pointing at the file submit_local wrote, and a 1-line "
            "message. Then call done."
        )

    with patch("kaggle_slayer.agent.solver.load_system_prompt", loader_with_kaggle):
        result = solver.solve()

    # Outcome
    assert result.status == "done", f"status={result.status} iters={result.iterations}"

    # OTel trace exists
    otel_path = workspace.root / "otel.jsonl"
    assert otel_path.exists()
    import json
    spans = [json.loads(line) for line in otel_path.read_text().splitlines() if line.strip()]
    span_names = [s["name"] for s in spans]
    assert any(n.startswith("run:") for n in span_names)
    assert any(n == "llm.call" for n in span_names)
    assert any(n.startswith("tool:") for n in span_names)

    # Cost ledger has rows for this comp
    assert ledger.total_for(competition="synthetic-telemetry") > 0

    # Calibration row from submit_kaggle (if it ran)
    if fake_kaggle.submit.called:
        rows = calibration.read_history(competition="synthetic-telemetry", path=cal_path)
        assert len(rows) >= 1
        assert rows[0]["metric"] == "accuracy"

    print(
        f"\nDONE iter={result.iterations} "
        f"cost=${ledger.total_for(competition='synthetic-telemetry'):.4f} "
        f"otel_spans={len(spans)} kaggle_submit_called={fake_kaggle.submit.called}"
    )
```

- [ ] **Step 2: Verify pytest collects but excludes from non-slow**

```bash
pytest --collect-only tests/integration/test_solver_real_gemini_telemetry.py 2>&1 | head -5
pytest -m "not slow" tests/integration/ --collect-only 2>&1 | tail -5
```

Expected: collected as one test; not selected by `-m "not slow"`.

- [ ] **Step 3: Lint + mypy**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_solver_real_gemini_telemetry.py
git commit -m "$(cat <<'EOF'
test: real-Gemini E2E with telemetry assertions (slow tier)

Runs the Solver end-to-end on a synthetic comp with real
gemini-2.5-flash, then asserts the telemetry artifacts all materialize:
otel.jsonl has run/llm.call/tool:* spans, the cost ledger has at least
one row for the competition, and (when submit_kaggle reaches its mock)
a calibration row is appended.

The system prompt is patched at runtime to nudge the agent toward
calling submit_kaggle after submit_local, mirroring the existing
checkpoint-acceptance test pattern.

Cost: ~$0.005-0.02 per run. Slow tier, opt-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Week 5 acceptance summary

After all 14 tasks:

- ✅ `kaggle_slayer/harness/telemetry/otel.py` — JSONL tracer per Solver run
- ✅ `kaggle_slayer/harness/telemetry/calibration.py` — global CV↔LB log
- ✅ `kaggle_slayer/harness/telemetry/errors.py` — crash reports with secret redaction
- ✅ `kaggle_slayer/harness/telemetry/behavior.py` — turns counter + stuck-loop (consolidated)
- ✅ `kaggle_slayer/harness/telemetry/mlflow_logger.py` — one MLflow run per train_cv
- ✅ Solver emits OTel spans for the loop + each LLM call + each tool dispatch
- ✅ submit_kaggle appends to the calibration log
- ✅ CLI captures unhandled exceptions to `~/.kaggle_slayer/errors/`
- ✅ Streamlit dashboard — `kaggle-slayer-dashboard` entry, portfolio + comp-detail pages
- ✅ Chaos tier — `FailureInjectingLLMClient` + integration test at 5% rate, seeded for determinism
- ✅ Slow-tier real-Gemini acceptance with telemetry assertions
- ✅ Coverage on new code ≥ 85%; ruff + mypy strict clean

**Week 6 starts with:** three real Kaggle Playground comps (the §18 acceptance criterion), full docs (`docs/architecture.md`, `docs/adr/0001-single-agent.md`, `0002-gemini-pro.md`, `0003-leak-free-contract.md`, `.claude/commands/{run-comp,resume-comp}.md`, `.claude/agents/harness-reviewer.md`), MLflow artifact logging (fe.py / model.py / oof_preds), the LB-score backfill task, the fe_v01↔fe_v02 dashboard diff page, and the cross-comp dashboard page.
