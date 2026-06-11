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
    # shutdown() is a no-op now; writes are sync.
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    assert path.exists()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # make_tracer writes a boundary marker; expect 1 (marker) + 1 (hello)
    assert len(records) == 2
    marker = next(r for r in records if r["name"].startswith("run:"))
    assert marker["status"] == "MARKER"
    assert marker["duration_ns"] == 0
    hello = next(r for r in records if r["name"] == "hello")
    assert hello["attributes"]["k"] == "v"
    assert hello["attributes"]["more"] == 1


def test_tracer_nests_child_spans(ws):
    tracer = otel.make_tracer(ws, run_name="solve")
    with tracer.start_span("outer"):
        with tracer.start_span("inner"):
            pass
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [r["name"] for r in records]
    assert "outer" in names and "inner" in names
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
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rec = next(r for r in records if r["name"] == "timed")
    # Duration in nanoseconds; >=10ms = 10_000_000 ns
    assert rec["duration_ns"] >= 10_000_000


def test_tracer_records_exception_status(ws):
    tracer = otel.make_tracer(ws, run_name="solve")
    with pytest.raises(RuntimeError, match="boom"):
        with tracer.start_span("failing"):
            raise RuntimeError("boom")
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rec = next(r for r in records if r["name"] == "failing")
    assert rec["status"] == "ERROR"
    assert "boom" in rec.get("error", "")


def test_set_attribute_after_span_exit_does_not_mutate_written_record(ws):
    """The JSONL record is sealed on exit; in-memory mutation has no
    observable effect.

    Concretely: the Span dataclass itself stays mutable after `__exit__`
    runs (we don't enforce `_sealed = True`), but the record has already
    been serialized to disk by then. Any later `set_attribute` call on the
    same Span instance updates only the live object — the appended JSONL
    line is unchanged.
    """
    tracer = otel.make_tracer(ws, run_name="solve")
    with tracer.start_span("sealed", attributes={"early": "yes"}) as span:
        pass
    # Span object is still alive; mutate after context exit.
    span.set_attribute("late", True)
    otel.shutdown()

    path = ws.root / "otel.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rec = next(r for r in records if r["name"] == "sealed")
    assert rec["attributes"].get("early") == "yes"
    assert "late" not in rec["attributes"]

def test_span_write_failure_does_not_crash(tmp_path):
    """Hard rule #6: an unwritable otel.jsonl must not abort the Solver loop
    (or mask the in-flight tool exception from a span's finally block)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be")
    t = otel.Tracer(file_path=blocker / "otel.jsonl", trace_id="t1")
    with t.start_span("solve.loop"):
        pass  # exiting the span writes; the failure must be swallowed
