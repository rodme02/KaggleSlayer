"""OpenTelemetry tracer with a JSONL file exporter.

Each call to `make_tracer(workspace, run_name=...)` returns a Tracer whose
spans append to `<workspace>/otel.jsonl`. Spans are nested via Python
context managers; parent/child relationships are recorded on the span's
`parent_span_id` field. Exceptions raised inside a span set status=ERROR
and record the error message.

We don't ship a full OTLP collector - the JSONL file is enough for the
dashboard and post-hoc debugging. If we ever need OTLP, swap the
processor in `_install_processor`.

Concurrency note: the tracer is single-process and single-threaded
(Solver runs sequentially). Two concurrent writers would interleave
records in `otel.jsonl`; no locking is added because YAGNI.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_slayer.harness.workspace import Workspace

_log = logging.getLogger(__name__)

_OTEL_FILENAME = "otel.jsonl"


@dataclass
class Span:
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


class Tracer:
    def __init__(self, file_path: Path, trace_id: str) -> None:
        self._file = file_path
        self._trace_id = trace_id
        self._stack: list[Span] = []

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        span = Span(
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

    def _write_marker(self, name: str) -> None:
        """Write a boundary marker record with duration_ns=0 and status=MARKER.

        Used by `make_tracer` to stamp a `run:<name>` record at the top of
        the file so traces are easy to locate. Distinct from a real span
        because no work was timed.
        """
        marker = Span(
            name=name,
            span_id=os.urandom(8).hex(),
            trace_id=self._trace_id,
            parent_span_id=None,
            started_ns=time.perf_counter_ns(),
            status="MARKER",
        )
        self._write(marker, duration_ns=0)

    def _write(self, span: Span, duration_ns: int) -> None:
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
        # Hard rule #6: tracing must never abort the solve. _write runs in a
        # span's `finally`, where a raise would also mask the tool's own
        # in-flight exception — swallow and log instead.
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with self._file.open("a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            _log.warning(
                "otel: failed to append span %r to %s; continuing",
                span.name, self._file, exc_info=True,
            )


def make_tracer(workspace: Workspace, *, run_name: str) -> Tracer:
    """Return a tracer that appends spans to <workspace>/otel.jsonl."""
    trace_id = os.urandom(16).hex()
    file_path = workspace.root / _OTEL_FILENAME
    tracer = Tracer(file_path=file_path, trace_id=trace_id)
    # Stamp a boundary marker so reading the file in order makes the trace
    # start obvious. Marker records have duration_ns=0 and status=MARKER.
    tracer._write_marker(name=f"run:{run_name}")
    return tracer


def shutdown() -> None:
    """No-op. Kept for API compatibility with future OTel SDK swaps.

    Writes are synchronous (each `_write` calls `flush()` + `fsync()`), so
    there is nothing to flush at shutdown.
    """
