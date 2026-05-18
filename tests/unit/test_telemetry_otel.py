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
    # make_tracer writes a synthetic `run:<name>` span; expect 1 (root) + 1 (hello)
    assert len(records) == 2
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
