"""OpenTelemetry tracer with a JSONL file exporter.

Each call to `make_tracer(workspace, run_name=...)` returns a Tracer whose
spans append to `<workspace>/otel.jsonl`. Spans are nested via Python
context managers; parent/child relationships are recorded on the span's
`parent_span_id` field. Exceptions raised inside a span set status=ERROR
and record the error message.

We don't ship a full OTLP collector - the JSONL file is enough for the
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
        except Exception as e:  # noqa: BLE001 - record then re-raise
            span.status = "ERROR"
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            ended = time.perf_counter_ns()
            duration = ended - span.started_ns
            self._stack.pop()
            self._write(span, duration)

    def _write(self, span: _Span, duration_ns: int) -> None:
        record: dict[str, Any] = {
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
