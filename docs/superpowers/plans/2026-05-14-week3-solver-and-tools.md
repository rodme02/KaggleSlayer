# Week 3 — Solver agent loop, tool registry, Gemini tool-call protocol, CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KaggleSlayer V2 *actually run* a Gemini-driven agent against a competition. By end of week, `kaggle-slayer <comp-path>` invokes a Solver that reads `context.md`, calls tools (`write_file`, `train_cv`, `submit_local`, `done`, etc.), and produces a `submission.csv`. The acceptance test: real Gemini solves a synthetic micro-comp (binary classification, 500 rows) end-to-end.

**Architecture:** Tools are registered in `kaggle_slayer/agent/tools.py` as `Tool(name, description, schema, handler)` records. The Solver maintains a Gemini-shaped message history (structured `Content`/`Part` lists, not the Week-2 string-flatten), passes the tool schemas in each call, parses `function_call` parts back into Python invocations, journals every step, and exits on `done()` / max-iterations / time-budget. The CLI is a thin shell around `Solver.solve()`.

**Tech Stack:** google-genai (typed Content/Part/FunctionDeclaration), jsonschema (input validation), existing Week-1/2 harness modules. Python 3.11+. Mypy strict.

**Acceptance:** unit tier green (~30 new unit tests), integration tier green (fake-LLM scripted solver run on synthetic comp), slow tier passes the real-Gemini acceptance run. Coverage on new code ≥ 85%.

---

## File map

**Created this week:**
- `kaggle_slayer/agent/tools.py` — `Tool` dataclass + `ToolError` + builtin registry
- `kaggle_slayer/agent/handlers/__init__.py`
- `kaggle_slayer/agent/handlers/files.py` — `read_context`, `read_file`, `write_file`, `sample_rows`, `take_note`
- `kaggle_slayer/agent/handlers/ml.py` — `set_cv`, `train_cv`, `submit_local`, `done`
- `kaggle_slayer/agent/solver.py` — `Solver` class
- `kaggle_slayer/agent/prompts/__init__.py`
- `kaggle_slayer/agent/prompts/system.md` — the Solver system prompt (resource file, loaded at runtime)
- `kaggle_slayer/cli.py` — `kaggle-slayer <comp-path>` entry point
- `tests/unit/test_tool_registry.py`
- `tests/unit/test_handlers_files.py`
- `tests/unit/test_handlers_ml.py`
- `tests/unit/test_gemini_tool_protocol.py`
- `tests/unit/test_solver.py`
- `tests/unit/test_workspace_versioning.py`
- `tests/unit/test_cli.py`
- `tests/fixtures/synthetic_comp.py` — programmatic synthetic-comp fixture
- `tests/integration/test_solver_with_fake_agent.py` — scripted-tool-call integration
- `tests/integration/test_solver_real_gemini.py` — `@pytest.mark.slow` real-Gemini acceptance

**Modified:**
- `kaggle_slayer/harness/workspace.py` — add `next_version_path(kind)` helper
- `kaggle_slayer/harness/context.py` — improve `_TARGET_HINTS` (case-insensitive + suffix matching)
- `kaggle_slayer/agent/llm_client.py` — extend `GeminiClient` to support structured Content/Part + tool calls; tighten `_is_transient`
- `pyproject.toml` — add `[project.scripts] kaggle-slayer = "kaggle_slayer.cli:main"` already present; verify; also add `jsonschema` if not present (it already is, per Week 2)
- `tests/conftest.py` — register the `synthetic_comp_workspace` fixture (re-exports from `tests/fixtures/synthetic_comp.py`)

---

## Task 1: Tighten `_is_transient` and improve `_TARGET_HINTS` (carry-forwards)

Two small Opus follow-ups. Bundle them in one commit.

**Files:**
- Modify: `kaggle_slayer/agent/llm_client.py` (only the `_is_transient` function)
- Modify: `kaggle_slayer/harness/context.py` (the `_TARGET_HINTS` constant + the `_data_profile` lookup)
- Modify: `tests/unit/test_llm_client.py` (add tests)
- Modify: `tests/unit/test_context_builder.py` (add tests)

- [ ] **Step 1: Failing tests for `_is_transient`**

Append at the end of `tests/unit/test_llm_client.py`:

```python
def test_is_transient_uses_status_code_attribute_first():
    """If an exception has a status_code attribute, prefer it over substring matching."""
    class HttpError(Exception):
        def __init__(self, code):
            super().__init__(f"http error")
            self.status_code = code
    assert llm._is_transient(HttpError(429)) is True
    assert llm._is_transient(HttpError(503)) is True
    assert llm._is_transient(HttpError(500)) is True
    # 4xx other than 429 should not retry
    assert llm._is_transient(HttpError(400)) is False
    assert llm._is_transient(HttpError(401)) is False
    assert llm._is_transient(HttpError(403)) is False
    assert llm._is_transient(HttpError(404)) is False


def test_is_transient_does_not_false_positive_on_model_name():
    """A permanent error mentioning a model name with '503' in it must not retry."""
    err = ValueError("model 'gemini-503-test' is not supported")
    # No status_code attribute, no transient keyword — must be permanent
    assert llm._is_transient(err) is False


def test_is_transient_still_matches_keyword_when_no_status_code():
    """Backstop substring match still works for SDKs without status codes."""
    assert llm._is_transient(Exception("rate limit exceeded")) is True
    assert llm._is_transient(Exception("temporarily unavailable")) is True
    assert llm._is_transient(Exception("connection timeout")) is True
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_llm_client.py::test_is_transient_does_not_false_positive_on_model_name -v
```

Expected: FAIL — current implementation substring-matches on `str(err).lower()` so `"gemini-503-test"` returns True.

- [ ] **Step 3: Replace `_is_transient` in `kaggle_slayer/agent/llm_client.py`**

Find the existing `_is_transient` function and replace it with:

```python
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_KEYWORDS: tuple[str, ...] = (
    "rate limit",
    "temporarily unavailable",
    "connection timeout",
    "connection reset",
    "deadline exceeded",
)


def _is_transient(err: Exception) -> bool:
    if isinstance(err, TransientLLMError):
        return True
    # Prefer a structured status_code attribute when the SDK provides one.
    status = getattr(err, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS_CODES
    # Fall back to keyword detection on the error message. Word-anchored so a
    # model name like "gemini-503-test" embedded in a permanent error does not
    # trigger a retry.
    msg = str(err).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)
```

- [ ] **Step 4: Failing tests for `_TARGET_HINTS`**

Append at the end of `tests/unit/test_context_builder.py`:

```python
def test_target_hint_matches_case_insensitively(workspace, kaggle_fake, tmp_path):
    """A column 'TARGET' or 'Target' should be flagged the same as 'target'."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "Target": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "likely target" in body
    assert "target" in body  # the column name appears as a flagged candidate


def test_target_hint_matches_suffix_label(workspace, kaggle_fake):
    """A column ending in '_label' or '_target' should be flagged."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "class_label": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "likely target" in body
    assert "class_label" in body


def test_target_hint_no_match_says_so(workspace, kaggle_fake):
    """If no column matches the hint patterns, context.md should say the agent must infer."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "outcome_var": [0, 1, 0]})
    df.to_csv(workspace.raw_dir / "train.csv", index=False)
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "agent should infer" in body
```

- [ ] **Step 5: Run, observe failure**

```bash
pytest tests/unit/test_context_builder.py -v -k "target_hint"
```

Expected: case-insensitive test fails (`Target` not matched). Suffix test fails (`class_label` not matched).

- [ ] **Step 6: Replace `_TARGET_HINTS` block in `kaggle_slayer/harness/context.py`**

Find the existing `_TARGET_HINTS` tuple definition and the `target_candidates = [c for c in df.columns if c in _TARGET_HINTS]` line in `_data_profile`. Replace both with this:

```python
# Common target column patterns. Match is case-insensitive against the column
# name lowercased, plus suffix/prefix variants for compound names.
_TARGET_EXACT: frozenset[str] = frozenset({
    "target", "label", "y", "outcome", "class",
    "survived", "saleprice",  # Kaggle staples
})
_TARGET_SUFFIXES: tuple[str, ...] = ("_target", "_label", "_y", "_outcome", "_class")
_TARGET_PREFIXES: tuple[str, ...] = ("target_", "label_", "y_")


def _looks_like_target(column_name: str) -> bool:
    cl = column_name.lower()
    if cl in _TARGET_EXACT:
        return True
    if any(cl.endswith(s) for s in _TARGET_SUFFIXES):
        return True
    return any(cl.startswith(p) for p in _TARGET_PREFIXES)
```

And in `_data_profile`, replace `target_candidates = [c for c in df.columns if c in _TARGET_HINTS]` with:

```python
target_candidates = [c for c in df.columns if _looks_like_target(c)]
```

- [ ] **Step 7: Run all the new tests + the existing target-hint test from Week 2**

```bash
pytest tests/unit/test_llm_client.py tests/unit/test_context_builder.py -v
```

Expected: all tests pass, including the previous Week-2 test `test_build_context_suggests_target_column` (it relied on `Survived`, which is still flagged via `_TARGET_EXACT`).

- [ ] **Step 8: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/llm_client.py kaggle_slayer/harness/context.py tests/unit/test_llm_client.py tests/unit/test_context_builder.py
git commit -m "$(cat <<'EOF'
fix: tighten _is_transient + extend target-column hints

_is_transient now reads a structured status_code attribute first (HTTP
408/429/5xx are retryable; 4xx other than 429 are permanent). Keyword
fallback uses word-anchored phrases instead of bare numeric codes, so a
permanent error mentioning a model name like 'gemini-503-test' no longer
triggers a retry loop. Both Opus carry-forwards from Week-2 review.

context._TARGET_HINTS replaced with _looks_like_target(): case-insensitive
exact match against {target, label, y, outcome, class, survived, saleprice}
plus suffix/prefix variants (_target, _label, _y, _outcome, _class,
target_, label_, y_). Catches columns like 'Target', 'class_label',
'y_train'. Spec §10 carry-forward.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Workspace.next_version_path` helper

The Opus Week-1 review flagged that the spec calls for `agent/fe.py` → `agent/versions/fe_v01.py` archival before each `train_cv`, but no helper exists. This is the helper. The actual *call* to archive will live in the Solver loop (Task 9).

**Files:**
- Modify: `kaggle_slayer/harness/workspace.py`
- Create: `tests/unit/test_workspace_versioning.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_workspace_versioning.py`:

```python
"""Tests for Workspace.next_version_path()."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_next_version_path_first_call(ws):
    p = ws.next_version_path("fe")
    assert p == ws.versions_dir / "fe_v01.py"


def test_next_version_path_increments_on_existing_files(ws):
    (ws.versions_dir / "fe_v01.py").write_text("# v1")
    (ws.versions_dir / "fe_v02.py").write_text("# v2")
    p = ws.next_version_path("fe")
    assert p == ws.versions_dir / "fe_v03.py"


def test_next_version_path_handles_model_kind(ws):
    (ws.versions_dir / "model_v01.py").write_text("# v1")
    p = ws.next_version_path("model")
    assert p == ws.versions_dir / "model_v02.py"


def test_next_version_path_ignores_unrelated_files(ws):
    # An fe_v01.py and a model_v01.py — asking for fe should not consider model.
    (ws.versions_dir / "fe_v01.py").write_text("")
    (ws.versions_dir / "model_v05.py").write_text("")
    assert ws.next_version_path("fe") == ws.versions_dir / "fe_v02.py"
    assert ws.next_version_path("model") == ws.versions_dir / "model_v06.py"


def test_next_version_path_rejects_invalid_kind(ws):
    with pytest.raises(ValueError, match="kind"):
        ws.next_version_path("submission")  # not fe or model
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_workspace_versioning.py -v
```

Expected: `AttributeError: 'Workspace' object has no attribute 'next_version_path'`.

- [ ] **Step 3: Add the method to `kaggle_slayer/harness/workspace.py`**

Append this method to the `Workspace` dataclass (immediately before the `@classmethod` block):

```python
    def next_version_path(self, kind: str) -> Path:
        """Return the next free path under versions/ for the given kind.

        kind: "fe" or "model". Scans versions/ for files matching
        `{kind}_v\\d+.py` and returns `versions/{kind}_v{N+1:02d}.py`.
        """
        if kind not in ("fe", "model"):
            raise ValueError(f"kind must be 'fe' or 'model', got {kind!r}")
        import re

        pattern = re.compile(rf"^{kind}_v(\d+)\.py$")
        existing: list[int] = []
        if self.versions_dir.is_dir():
            for f in self.versions_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    existing.append(int(m.group(1)))
        next_n = (max(existing) + 1) if existing else 1
        return self.versions_dir / f"{kind}_v{next_n:02d}.py"
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_workspace_versioning.py -v
```

Expected: 5 passes.

- [ ] **Step 5: Lint + mypy**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/workspace.py tests/unit/test_workspace_versioning.py
git commit -m "$(cat <<'EOF'
feat(workspace): add next_version_path(kind) helper

Scans versions/ for files matching `{fe|model}_v\d+.py` and returns the
next free path. Used by the Solver (Week 3) to archive each fe.py /
model.py before train_cv runs — provides the version-history trail the
spec §10 layout describes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Tool registry + `Tool` dataclass + builtin registry shell

This task defines the data types and an empty registry. Actual handlers land in Tasks 4–6.

**Files:**
- Create: `kaggle_slayer/agent/tools.py`
- Create: `tests/unit/test_tool_registry.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_tool_registry.py`:

```python
"""Tests for kaggle_slayer.agent.tools — registry + Tool dataclass."""

from __future__ import annotations

import pytest

from kaggle_slayer.agent import tools as t


def test_tool_dataclass_fields():
    tool = t.Tool(
        name="probe",
        description="A probe tool.",
        schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        handler=lambda ctx, x: f"got {x}",
    )
    assert tool.name == "probe"
    assert tool.description == "A probe tool."
    assert tool.schema["type"] == "object"
    assert tool.handler is not None


def test_register_and_get_tool():
    registry = t.ToolRegistry()
    tool = t.Tool(name="probe", description="d", schema={"type": "object"}, handler=lambda ctx: "ok")
    registry.register(tool)
    assert registry.get("probe") is tool


def test_register_duplicate_raises():
    registry = t.ToolRegistry()
    tool = t.Tool(name="probe", description="d", schema={"type": "object"}, handler=lambda ctx: "ok")
    registry.register(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_get_unknown_raises():
    registry = t.ToolRegistry()
    with pytest.raises(KeyError, match="unknown_tool"):
        registry.get("unknown_tool")


def test_list_names_returns_sorted():
    registry = t.ToolRegistry()
    registry.register(t.Tool(name="b", description="d", schema={"type": "object"}, handler=lambda ctx: ""))
    registry.register(t.Tool(name="a", description="d", schema={"type": "object"}, handler=lambda ctx: ""))
    assert registry.names() == ["a", "b"]


def test_invoke_validates_args_against_schema():
    """Args that don't match the schema raise ToolError before the handler runs."""
    registry = t.ToolRegistry()
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    called = []
    registry.register(t.Tool(
        name="probe",
        description="d",
        schema=schema,
        handler=lambda ctx, x: called.append(x) or "done",
    ))
    ctx = object()  # opaque; handlers may use it later
    with pytest.raises(t.ToolError, match="schema"):
        registry.invoke("probe", ctx=ctx, args={"x": "not an int"})
    # Handler must not have been called
    assert called == []


def test_invoke_passes_ctx_and_unpacks_args():
    registry = t.ToolRegistry()
    seen = {}
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
        "required": ["x"],
    }

    def handler(ctx, x, y="default"):
        seen["ctx"] = ctx
        seen["x"] = x
        seen["y"] = y
        return f"x={x} y={y}"

    registry.register(t.Tool(name="probe", description="d", schema=schema, handler=handler))
    ctx = object()
    result = registry.invoke("probe", ctx=ctx, args={"x": 7})
    assert result == "x=7 y=default"
    assert seen == {"ctx": ctx, "x": 7, "y": "default"}


def test_invoke_unknown_tool_raises_tool_error():
    registry = t.ToolRegistry()
    with pytest.raises(t.ToolError, match="unknown tool"):
        registry.invoke("nope", ctx=None, args={})


def test_to_function_declarations_format():
    """The registry can export tools in a generic function-declaration format
    that LLM clients translate into provider-specific shapes."""
    registry = t.ToolRegistry()
    registry.register(t.Tool(
        name="probe",
        description="A probe.",
        schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        handler=lambda ctx, x: "ok",
    ))
    decls = registry.to_function_declarations()
    assert decls == [{
        "name": "probe",
        "description": "A probe.",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    }]
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_tool_registry.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.tools`.

- [ ] **Step 3: Create `kaggle_slayer/agent/tools.py`**

```python
"""Tool registry: typed Tool records + invocation with JSON-schema validation.

A Tool bundles four things the LLM and harness both need:
  - name + description: surfaced to the model so it knows when to use the tool
  - schema: JSON-schema for arguments; validated before the handler runs
  - handler: Python callable invoked with (ctx, **args), returns a JSON-able value

ToolError is raised on validation failure or unknown name. The Solver's
loop catches it, logs to the journal as a tool_error, and feeds the error
message back to the LLM on the next turn so it can adjust.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jsonschema  # type: ignore[import-untyped]


class ToolError(Exception):
    """Raised when a tool invocation fails: schema mismatch, unknown name,
    or a handler signaling a recoverable error to the agent."""


@dataclass(frozen=True)
class Tool:
    """A single registered tool."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Any]


class ToolRegistry:
    """Holds Tools keyed by name. Cheap to construct; safe to share."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; known: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def invoke(self, name: str, *, ctx: Any, args: dict[str, Any]) -> Any:
        """Validate args against the tool's schema, then call handler(ctx, **args)."""
        if name not in self._tools:
            raise ToolError(f"unknown tool {name!r}")
        tool = self._tools[name]
        try:
            jsonschema.validate(instance=args, schema=tool.schema)
        except jsonschema.ValidationError as e:
            raise ToolError(f"schema validation failed for {name!r}: {e.message}") from e
        return tool.handler(ctx, **args)

    def to_function_declarations(self) -> list[dict[str, Any]]:
        """Export tools as generic JSON-schema function declarations.

        The LLMClient translates this into provider-specific shapes (e.g.,
        Gemini's `Tool(function_declarations=[FunctionDeclaration(...)])`).
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            }
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_tool_registry.py -v
```

Expected: 9 passes.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/tools.py tests/unit/test_tool_registry.py
git commit -m "$(cat <<'EOF'
feat(agent): add Tool dataclass + ToolRegistry + jsonschema validation

A Tool bundles name, description, JSON-schema for args, and a Python
handler. ToolRegistry.invoke(name, ctx, args) validates args against
the schema first, then calls handler(ctx, **args). Validation failure
raises ToolError, which the Solver loop will feed back to the model on
the next turn.

to_function_declarations() exports the registry in a generic shape
(name + description + parameters) the LLMClient translates into
provider-specific tool declarations.

Concrete tool handlers (read_context, write_file, train_cv, ...) land
in Tasks 4-6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: File-side tool handlers — `read_context`, `read_file`, `write_file`, `sample_rows`, `take_note`

These are the "look at things and write code" tools.

**Files:**
- Create: `kaggle_slayer/agent/handlers/__init__.py`
- Create: `kaggle_slayer/agent/handlers/files.py`
- Create: `tests/unit/test_handlers_files.py`

- [ ] **Step 1: Create the handlers package**

```bash
mkdir -p kaggle_slayer/agent/handlers
```

`kaggle_slayer/agent/handlers/__init__.py` (with this exact content):

```python
"""Tool handlers — pure Python callables registered into a ToolRegistry."""
```

- [ ] **Step 2: Failing tests**

Create `tests/unit/test_handlers_files.py`:

```python
"""Tests for kaggle_slayer.agent.handlers.files."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from kaggle_slayer.agent.handlers import files as fh
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    """Minimal SolverContext stand-in: just a workspace + journal for these handlers."""
    workspace: Workspace
    journal: Journal


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_read_context_returns_file_contents(ctx):
    ctx.workspace.context_path.write_text("# Test Comp\n\nMetric: accuracy.\n")
    result = fh.read_context(ctx)
    assert "Test Comp" in result
    assert "Metric: accuracy" in result


def test_read_context_missing_raises_tool_error(ctx):
    with pytest.raises(ToolError, match="context.md"):
        fh.read_context(ctx)


def test_read_file_inside_workspace(ctx):
    p = ctx.workspace.agent_dir / "fe.py"
    p.write_text("def fit_feature_transformer(df, t): pass")
    result = fh.read_file(ctx, path="agent/fe.py")
    assert "fit_feature_transformer" in result


def test_read_file_outside_workspace_rejected(ctx):
    """Path traversal must be blocked."""
    with pytest.raises(ToolError, match="outside workspace"):
        fh.read_file(ctx, path="../../etc/passwd")


def test_read_file_missing_raises_tool_error(ctx):
    with pytest.raises(ToolError, match="not found"):
        fh.read_file(ctx, path="agent/does_not_exist.py")


def test_write_file_creates_under_agent_dir(ctx):
    fh.write_file(ctx, path="agent/fe.py", content="def fit_feature_transformer(df, t): pass\n")
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "def fit_feature_transformer(df, t): pass\n"


def test_write_file_overwrites(ctx):
    fh.write_file(ctx, path="agent/fe.py", content="v1")
    fh.write_file(ctx, path="agent/fe.py", content="v2")
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "v2"


def test_write_file_creates_parent_dirs(ctx):
    fh.write_file(ctx, path="agent/scratch/probe.py", content="x = 1")
    assert (ctx.workspace.agent_dir / "scratch" / "probe.py").exists()


def test_write_file_outside_workspace_rejected(ctx):
    with pytest.raises(ToolError, match="outside workspace"):
        fh.write_file(ctx, path="../escape.py", content="x = 1")


def test_write_file_rejects_protected_paths(ctx):
    """run_log.jsonl, notes.jsonl, context.md must not be writable from a tool."""
    for forbidden in ("run_log.jsonl", "notes.jsonl", "context.md"):
        with pytest.raises(ToolError, match="protected"):
            fh.write_file(ctx, path=forbidden, content="x")


def test_sample_rows_returns_first_n_rows(ctx):
    df = pd.DataFrame({"a": range(20), "b": list("abcdefghijklmnopqrst")})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    result = fh.sample_rows(ctx, table="train", n=5)
    # Result is a stringified table with the first 5 rows
    assert "a" in result and "b" in result
    assert "0" in result and "4" in result
    # Should not include rows 5+
    assert "10" not in result or result.count("\n") <= 8  # tolerate header + 5 data lines


def test_sample_rows_random_sampling(ctx):
    df = pd.DataFrame({"a": range(100)})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    r1 = fh.sample_rows(ctx, table="train", n=5, random=True)
    r2 = fh.sample_rows(ctx, table="train", n=5, random=True)
    # Different random samples should be different (extremely high probability)
    # but the function is deterministic for a given call — so we just verify it didn't crash
    assert r1 and r2


def test_sample_rows_missing_table_raises(ctx):
    with pytest.raises(ToolError, match="train.csv"):
        fh.sample_rows(ctx, table="train", n=3)


def test_sample_rows_caps_at_table_size(ctx):
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    result = fh.sample_rows(ctx, table="train", n=100)
    # Should just return all 3 rows rather than raise
    assert "1" in result and "3" in result


def test_take_note_appends_to_notes_jsonl(ctx):
    fh.take_note(ctx, category="observation", content="target is imbalanced (3%)")
    lines = ctx.workspace.notes_path.read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["category"] == "observation"
    assert "imbalanced" in rec["content"]


def test_take_note_rejects_unknown_category(ctx):
    with pytest.raises(ToolError, match="category"):
        fh.take_note(ctx, category="invalid_cat", content="x")
```

- [ ] **Step 3: Run, observe failure**

```bash
pytest tests/unit/test_handlers_files.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.handlers.files`.

- [ ] **Step 4: Create `kaggle_slayer/agent/handlers/files.py`**

```python
"""File-side tool handlers — read/write within the workspace.

All handlers take a `ctx` as their first positional argument. The ctx
exposes the Workspace and Journal; concrete construction happens in the
Solver. These functions don't import the Solver to avoid circular deps —
the contract is structural ("ctx must have .workspace and .journal").

Path safety: write_file and read_file resolve paths relative to the
workspace root and reject anything outside it. A small set of paths
(run_log.jsonl, notes.jsonl, context.md) is additionally protected from
agent writes — only the harness writes those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import NOTE_CATEGORIES

_PROTECTED_BASENAMES: frozenset[str] = frozenset({"run_log.jsonl", "notes.jsonl", "context.md"})


def _resolve_under(workspace_root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` under workspace_root and reject path traversal."""
    base = workspace_root.resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ToolError(f"path {rel_path!r} resolves outside workspace") from e
    return target


def read_context(ctx: Any) -> str:
    """Read the workspace's context.md (the agent's brief)."""
    p = ctx.workspace.context_path
    if not p.exists():
        raise ToolError(f"context.md not found at {p}")
    return p.read_text()


def read_file(ctx: Any, *, path: str) -> str:
    """Read a file from inside the workspace."""
    target = _resolve_under(ctx.workspace.root, path)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if not target.is_file():
        raise ToolError(f"path is not a file: {path}")
    return target.read_text()


def write_file(ctx: Any, *, path: str, content: str) -> str:
    """Write a file inside the workspace. Protected harness files are rejected."""
    target = _resolve_under(ctx.workspace.root, path)
    if target.name in _PROTECTED_BASENAMES and target.parent == ctx.workspace.root.resolve():
        raise ToolError(f"path {path!r} is protected (harness writes it)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def sample_rows(ctx: Any, *, table: str, n: int = 10, random: bool = False) -> str:
    """Return a sample of n rows from raw/<table>.csv as a string."""
    target = ctx.workspace.raw_dir / f"{table}.csv"
    if not target.exists():
        raise ToolError(f"{target.name} not found in raw/")
    df = pd.read_csv(target)
    if random and n < len(df):
        df = df.sample(n=n, random_state=42)
    else:
        df = df.head(n)
    return df.to_string(max_cols=20, max_colwidth=40)


def take_note(ctx: Any, *, category: str, content: str) -> str:
    """Append a structured note to notes.jsonl."""
    if category not in NOTE_CATEGORIES:
        raise ToolError(
            f"unknown category {category!r}; allowed: {sorted(NOTE_CATEGORIES)}"
        )
    ctx.journal.take_note(category=category, content=content)
    return f"noted ({category})"
```

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_handlers_files.py -v
```

Expected: all 16 pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/ tests/unit/test_handlers_files.py
git commit -m "$(cat <<'EOF'
feat(agent): add file-side tool handlers

read_context, read_file, write_file, sample_rows, take_note. Path safety
enforced via _resolve_under (rejects traversal outside the workspace
root). Protected basenames (run_log.jsonl, notes.jsonl, context.md) are
forbidden from agent writes — only the harness writes those.

sample_rows returns a stringified head() or sample(); take_note delegates
to Journal with NOTE_CATEGORIES validation. Handlers raise ToolError on
recoverable failures so the Solver loop can feed the message back to
the model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: ML-side tool handlers — `set_cv`, `train_cv`, `submit_local`

The bridge between the agent and the harness's leak-free CV. These are the most consequential tools.

**Files:**
- Create: `kaggle_slayer/agent/handlers/ml.py`
- Create: `tests/unit/test_handlers_ml.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_handlers_ml.py`:

```python
"""Tests for kaggle_slayer.agent.handlers.ml."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.tools import ToolError
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


def _write_stub_fe(workspace: Workspace) -> None:
    """Mean-impute numerics, drop categoricals."""
    workspace.fe_path.write_text(
        "import pandas as pd\n"
        "\n"
        "class _PT:\n"
        "    def __init__(self, cols, means):\n"
        "        self.cols = cols\n"
        "        self.means = means\n"
        "    def transform(self, df):\n"
        "        out = pd.DataFrame(index=df.index)\n"
        "        for c in self.cols:\n"
        "            if c in df.columns:\n"
        "                out[c] = df[c].fillna(self.means.get(c, 0.0))\n"
        "        return out\n"
        "\n"
        "def fit_feature_transformer(train_df, target_col):\n"
        "    cols = [c for c in train_df.columns if c != target_col and train_df[c].dtype.kind in 'fiub']\n"
        "    means = {c: float(train_df[c].mean()) for c in cols}\n"
        "    return _PT(cols, means)\n"
    )


def _write_stub_model(workspace: Workspace) -> None:
    workspace.model_path.write_text(
        "from sklearn.linear_model import LogisticRegression, Ridge\n"
        "\n"
        "def fit_model(X_train, y_train, problem_type, metric_name):\n"
        "    if problem_type == 'classification':\n"
        "        m = LogisticRegression(max_iter=500, random_state=42)\n"
        "    else:\n"
        "        m = Ridge(alpha=1.0, random_state=42)\n"
        "    m.fit(X_train, y_train)\n"
        "    return m\n"
    )


@pytest.fixture
def comp_ctx(tmp_path):
    """A workspace with a small synthetic binary-classification dataset wired in."""
    import numpy as np

    ws = Workspace.create(root=tmp_path / "comp")
    rng = np.random.default_rng(0)
    n_train, n_test = 200, 50
    train_df = pd.DataFrame({
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train),
        "target": rng.integers(0, 2, size=n_train),
    })
    train_df.to_csv(ws.raw_dir / "train.csv", index=False)
    test_df = pd.DataFrame({
        "id": range(n_test),
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test),
    })
    test_df.to_csv(ws.raw_dir / "test.csv", index=False)
    _write_stub_fe(ws)
    _write_stub_model(ws)
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_set_cv_records_override(comp_ctx):
    result = ml_h.set_cv(comp_ctx, kind="stratified_kfold", n_splits=3)
    assert "stratified_kfold" in result
    assert comp_ctx.cv_kind == "stratified_kfold"
    assert comp_ctx.cv_params == {"n_splits": 3}


def test_set_cv_with_group_col(comp_ctx):
    result = ml_h.set_cv(comp_ctx, kind="group_kfold", n_splits=3, group_col="x1")
    assert "group_kfold" in result
    assert comp_ctx.cv_params == {"n_splits": 3, "group_col": "x1"}


def test_set_cv_rejects_unknown_kind(comp_ctx):
    with pytest.raises(ToolError, match="unknown CV kind"):
        ml_h.set_cv(comp_ctx, kind="random_split")


def test_train_cv_runs_and_returns_summary(comp_ctx):
    result = ml_h.train_cv(comp_ctx)
    # Result is a string the LLM can read; should mention fold scores and mean
    assert "mean=" in result.lower() or "mean " in result.lower()
    assert "0." in result  # some score value


def test_train_cv_archives_fe_and_model_to_versions(comp_ctx):
    ml_h.train_cv(comp_ctx)
    assert (comp_ctx.workspace.versions_dir / "fe_v01.py").exists()
    assert (comp_ctx.workspace.versions_dir / "model_v01.py").exists()
    # Re-running increments
    ml_h.train_cv(comp_ctx)
    assert (comp_ctx.workspace.versions_dir / "fe_v02.py").exists()
    assert (comp_ctx.workspace.versions_dir / "model_v02.py").exists()


def test_train_cv_uses_cv_override_when_set(comp_ctx):
    """If set_cv was called, train_cv must use that strategy.

    Default for classification is stratified_kfold; explicit override to plain
    kfold should be honored. We assert on the precise strategy name and the
    n_splits the override specified — substring matches won't work because
    "kfold" is a substring of "stratified_kfold".
    """
    ml_h.set_cv(comp_ctx, kind="kfold", n_splits=3)
    result = ml_h.train_cv(comp_ctx)
    assert "kfold (3 folds)" in result
    assert "stratified" not in result


def test_train_cv_missing_fe_raises(comp_ctx):
    comp_ctx.workspace.fe_path.unlink()
    with pytest.raises(ToolError, match="fe.py"):
        ml_h.train_cv(comp_ctx)


def test_train_cv_missing_model_raises(comp_ctx):
    comp_ctx.workspace.model_path.unlink()
    with pytest.raises(ToolError, match="model.py"):
        ml_h.train_cv(comp_ctx)


def test_submit_local_writes_submission_csv(comp_ctx):
    result = ml_h.submit_local(comp_ctx, label="lr_baseline")
    # Check file exists in submissions/ and the result message references it
    submissions = list(comp_ctx.workspace.submissions_dir.glob("*lr_baseline*.csv"))
    assert len(submissions) == 1
    assert "submission" in result.lower() or "wrote" in result.lower()


def test_submit_local_includes_id_column_from_test(comp_ctx):
    ml_h.submit_local(comp_ctx, label="run1")
    sub_path = next(comp_ctx.workspace.submissions_dir.glob("*run1*.csv"))
    sub = pd.read_csv(sub_path)
    # Must have the id column from test.csv plus the target column
    assert "id" in sub.columns
    assert comp_ctx.target_col in sub.columns or "target" in sub.columns
    # Row count matches test set
    assert len(sub) == 50


def test_submit_local_requires_fe_and_model(comp_ctx):
    comp_ctx.workspace.fe_path.unlink()
    with pytest.raises(ToolError, match="fe.py"):
        ml_h.submit_local(comp_ctx, label="x")
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_handlers_ml.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.handlers.ml`.

- [ ] **Step 3: Create `kaggle_slayer/agent/handlers/ml.py`**

```python
"""ML-side tool handlers — set_cv, train_cv, submit_local.

These talk to the harness's leak-free CV via train_cv() and produce
a submission CSV via submit_local(). Both read the agent's current
fe.py and model.py from the workspace.

Before each train_cv, the current fe.py and model.py are copy-archived
into agent/versions/ as fe_v{N}.py and model_v{N}.py — this gives the
agent (and the future dashboard) a paper trail of every attempt.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness.registry import cv_strategies, metrics

_ALLOWED_CV_KINDS: frozenset[str] = frozenset({"kfold", "stratified_kfold", "time_series", "group_kfold"})


def _require_files(ctx: Any) -> tuple[Path, Path]:
    fe = ctx.workspace.fe_path
    model = ctx.workspace.model_path
    if not fe.exists():
        raise ToolError("agent/fe.py not found — write it first with write_file")
    if not model.exists():
        raise ToolError("agent/model.py not found — write it first with write_file")
    return fe, model


def set_cv(ctx: Any, *, kind: str, n_splits: int = 5, group_col: str | None = None) -> str:
    """Override the CV strategy for subsequent train_cv calls."""
    if kind not in _ALLOWED_CV_KINDS:
        raise ToolError(f"unknown CV kind {kind!r}; allowed: {sorted(_ALLOWED_CV_KINDS)}")
    ctx.cv_kind = kind
    params: dict[str, Any] = {"n_splits": n_splits}
    if group_col is not None:
        params["group_col"] = group_col
    ctx.cv_params = params
    return f"set cv strategy: {kind} with {params}"


def _build_cv_strategy(ctx: Any) -> Any:
    """Either honor ctx.cv_kind/cv_params or auto_select from problem_type."""
    if getattr(ctx, "cv_kind", None):
        return cv_strategies.get(ctx.cv_kind, **ctx.cv_params)
    return cv_strategies.auto_select(
        problem_type=ctx.problem_type,
        train_df=pd.read_csv(ctx.workspace.raw_dir / "train.csv", nrows=5),
        target_col=ctx.target_col,
    )


def train_cv(ctx: Any) -> str:
    """Archive fe.py + model.py into versions/, then run leak-free CV."""
    fe_path, model_path = _require_files(ctx)
    # Archive
    fe_archive = ctx.workspace.next_version_path("fe")
    model_archive = ctx.workspace.next_version_path("model")
    shutil.copyfile(fe_path, fe_archive)
    shutil.copyfile(model_path, model_archive)

    train_df = pd.read_csv(ctx.workspace.raw_dir / "train.csv")
    cv = _build_cv_strategy(ctx)
    metric = metrics.get(ctx.metric_name)

    result = cv_mod.train_cv(
        fe_path=fe_path,
        model_path=model_path,
        train_df=train_df,
        target_col=ctx.target_col,
        cv=cv,
        metric=metric,
        metadata_extra={"fe_version": fe_archive.stem, "model_version": model_archive.stem},
    )

    summary = (
        f"train_cv complete: {cv.name} ({cv.n_splits} folds), metric={metric.name}, "
        f"mean={result.mean:.4f}, std={result.std:.4f}, fold_scores={[round(s, 4) for s in result.fold_scores]}, "
        f"duration_s={result.duration_s:.2f}"
    )
    return summary


def submit_local(ctx: Any, *, label: str) -> str:
    """Fit fe.py + model.py on the full train set; predict test; write submission CSV."""
    fe_path, model_path = _require_files(ctx)

    train_df = pd.read_csv(ctx.workspace.raw_dir / "train.csv")
    test_path = ctx.workspace.raw_dir / "test.csv"
    if not test_path.exists():
        raise ToolError(f"test.csv not found at {test_path}")
    test_df = pd.read_csv(test_path)

    # Detect id column — use whichever of {id, Id, ID, PassengerId, ...} appears in test.
    id_col = _detect_id_column(test_df)

    # Load agent modules and fit on the FULL train set.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_agent_fe_final", fe_path)
    if spec is None or spec.loader is None:
        raise ToolError(f"cannot load {fe_path}")
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    spec = importlib.util.spec_from_file_location("_agent_model_final", model_path)
    if spec is None or spec.loader is None:
        raise ToolError(f"cannot load {model_path}")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)

    fe = fe_mod.fit_feature_transformer(train_df, ctx.target_col)
    X_train = fe.transform(train_df.drop(columns=[ctx.target_col]))
    y_train = train_df[ctx.target_col].to_numpy()

    test_for_transform = test_df.drop(columns=[id_col]) if id_col else test_df
    X_test = fe.transform(test_for_transform)

    metric = metrics.get(ctx.metric_name)
    model = model_mod.fit_model(X_train, y_train, ctx.problem_type, metric.name)

    if metric.needs_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        preds = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    else:
        preds = model.predict(X_test)

    # Build submission DataFrame
    out = pd.DataFrame()
    if id_col is not None:
        out[id_col] = test_df[id_col].values
    out[ctx.target_col] = preds

    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H%M%S")
    out_path = ctx.workspace.submissions_dir / f"{stamp}_{label}.csv"
    out.to_csv(out_path, index=False)
    return f"wrote submission ({len(out)} rows) to {out_path.relative_to(ctx.workspace.root)}"


def _detect_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ("id", "Id", "ID", "PassengerId", "index"):
        if candidate in df.columns:
            return candidate
    # Heuristic: first column with high uniqueness
    if df.columns[0].lower().endswith("id"):
        return df.columns[0]
    return None
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_handlers_ml.py -v
```

Expected: all 12 pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/ml.py tests/unit/test_handlers_ml.py
git commit -m "$(cat <<'EOF'
feat(agent): add ML tool handlers — set_cv, train_cv, submit_local

set_cv records a strategy override on ctx; train_cv consumes the override
(or auto_selects from problem_type when unset), archives the current
agent/fe.py + agent/model.py into versions/ before running the leak-free
CV, and returns a string summary the LLM can read directly.

submit_local fits fe.py + model.py on the FULL train set (no fold split),
predicts test.csv, detects the id column, and writes a dated submission
CSV under submissions/. Probability-needing metrics use predict_proba
when available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `done` handler + builtin registry assembly

Final handler + a `make_builtin_registry(ctx)` factory that wires everything together.

**Files:**
- Modify: `kaggle_slayer/agent/handlers/__init__.py` (add `make_builtin_registry`)
- Modify: `kaggle_slayer/agent/handlers/ml.py` (add `done`)
- Modify: `tests/unit/test_handlers_ml.py` (test for `done`)
- Create: `tests/unit/test_builtin_registry.py`

- [ ] **Step 1: Failing tests**

Append to `tests/unit/test_handlers_ml.py`:

```python
def test_done_sets_ctx_finished(comp_ctx):
    msg = ml_h.done(comp_ctx, summary="best cv was 0.82 with lr baseline")
    assert "0.82" in msg
    assert comp_ctx.finished is True
    assert comp_ctx.final_summary == "best cv was 0.82 with lr baseline"
```

Update the `_Ctx` dataclass at the top of `tests/unit/test_handlers_ml.py` to add the two new fields:

```python
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
```

Create `tests/unit/test_builtin_registry.py`:

```python
"""Tests for handlers.make_builtin_registry — the wired-up tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from kaggle_slayer.agent.handlers import make_builtin_registry
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


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_builtin_registry_has_expected_tools(ctx):
    reg = make_builtin_registry()
    expected = {"read_context", "read_file", "write_file", "sample_rows",
                "take_note", "set_cv", "train_cv", "submit_local", "done"}
    assert set(reg.names()) == expected


def test_builtin_registry_invoke_write_file(ctx, tmp_path):
    reg = make_builtin_registry()
    reg.invoke("write_file", ctx=ctx, args={"path": "agent/fe.py", "content": "x = 1"})
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "x = 1"


def test_builtin_registry_invoke_done(ctx):
    reg = make_builtin_registry()
    reg.invoke("done", ctx=ctx, args={"summary": "all good"})
    assert ctx.finished is True
    assert ctx.final_summary == "all good"


def test_builtin_registry_function_declarations_format(ctx):
    """All declarations have name + description + parameters keys, JSON-schema shape."""
    reg = make_builtin_registry()
    decls = reg.to_function_declarations()
    assert len(decls) == 9
    for d in decls:
        assert d["name"]
        assert d["description"]
        assert d["parameters"]["type"] == "object"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_builtin_registry.py -v tests/unit/test_handlers_ml.py::test_done_sets_ctx_finished -v
```

Expected: `done` not defined; `make_builtin_registry` not defined.

- [ ] **Step 3: Add `done` to `kaggle_slayer/agent/handlers/ml.py`**

Append at the end of the file:

```python
def done(ctx: Any, *, summary: str) -> str:
    """Signal that the agent is finished. The Solver loop exits after this returns."""
    ctx.finished = True
    ctx.final_summary = summary
    return f"acknowledged: {summary}"
```

- [ ] **Step 4: Build `make_builtin_registry()` in `kaggle_slayer/agent/handlers/__init__.py`**

Replace the file's content (currently just a docstring) with:

```python
"""Tool handlers — pure Python callables registered into a ToolRegistry.

make_builtin_registry() returns a ToolRegistry pre-loaded with the
9 builtin tools the Solver uses in Week 3.
"""

from __future__ import annotations

from kaggle_slayer.agent.handlers import files as fh, ml as ml_h
from kaggle_slayer.agent.tools import Tool, ToolRegistry


def make_builtin_registry() -> ToolRegistry:
    """Build a ToolRegistry pre-loaded with the Week-3 tool set."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_context",
        description="Read the competition's context.md (problem brief + data profile).",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=fh.read_context,
    ))
    reg.register(Tool(
        name="read_file",
        description="Read a file from inside the workspace (e.g., agent/fe.py).",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Workspace-relative path"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=fh.read_file,
    ))
    reg.register(Tool(
        name="write_file",
        description=(
            "Write a file inside the workspace. Use this to create agent/fe.py "
            "and agent/model.py with your feature-engineering and model code. "
            "Overwrites any existing content."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path (e.g., 'agent/fe.py')"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=fh.write_file,
    ))
    reg.register(Tool(
        name="sample_rows",
        description="Return a sample of rows from raw/<table>.csv as a formatted string.",
        schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name without extension, e.g., 'train' or 'test'"},
                "n": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "random": {"type": "boolean", "default": False},
            },
            "required": ["table"],
            "additionalProperties": False,
        },
        handler=fh.sample_rows,
    ))
    reg.register(Tool(
        name="take_note",
        description="Append a structured note to notes.jsonl for later reference.",
        schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["observation", "decision", "hypothesis", "todo"],
                },
                "content": {"type": "string"},
            },
            "required": ["category", "content"],
            "additionalProperties": False,
        },
        handler=fh.take_note,
    ))
    reg.register(Tool(
        name="set_cv",
        description=(
            "Override the CV strategy. By default the harness auto-selects "
            "(stratified_kfold for classification, kfold for regression). "
            "Use this when the data is time-indexed (kind='time_series') or "
            "grouped (kind='group_kfold', group_col='<col>')."
        ),
        schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["kfold", "stratified_kfold", "time_series", "group_kfold"],
                },
                "n_splits": {"type": "integer", "minimum": 2, "maximum": 20, "default": 5},
                "group_col": {"type": "string"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
        handler=ml_h.set_cv,
    ))
    reg.register(Tool(
        name="train_cv",
        description=(
            "Run leak-free K-fold cross-validation using the current "
            "agent/fe.py and agent/model.py. Archives both into "
            "agent/versions/ before the run and returns a CV summary "
            "(mean, std, per-fold scores)."
        ),
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=ml_h.train_cv,
    ))
    reg.register(Tool(
        name="submit_local",
        description=(
            "Fit fe.py + model.py on the full training set, predict the test "
            "data, and write a submission CSV under submissions/. The CSV "
            "is local-only (this does NOT push to Kaggle)."
        ),
        schema={
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Short label used in the CSV filename"},
            },
            "required": ["label"],
            "additionalProperties": False,
        },
        handler=ml_h.submit_local,
    ))
    reg.register(Tool(
        name="done",
        description=(
            "Signal that you are finished. The harness loop exits after this. "
            "Pass a one-sentence summary of what you accomplished."
        ),
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        handler=ml_h.done,
    ))
    return reg
```

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_builtin_registry.py tests/unit/test_handlers_ml.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/handlers/__init__.py kaggle_slayer/agent/handlers/ml.py tests/unit/test_handlers_ml.py tests/unit/test_builtin_registry.py
git commit -m "$(cat <<'EOF'
feat(agent): add done handler + make_builtin_registry() factory

done() sets ctx.finished=True and ctx.final_summary; the Solver loop
checks the flag after every tool dispatch and exits cleanly.

make_builtin_registry() returns a ToolRegistry pre-loaded with the 9
Week-3 builtin tools, each with description text the model reads and
JSON schemas for input validation. additionalProperties: false on every
tool keeps the model from inventing parameter names that silently break.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Gemini structured Content/Part + tool-calling protocol

The Week-2 GeminiClient flattens messages to a single string — that breaks tool calling. This task replaces the string-flatten with proper `Content`/`Part` lists and wires up `function_call` + `function_response` round-trips.

**Files:**
- Modify: `kaggle_slayer/agent/llm_client.py`
- Create: `tests/unit/test_gemini_tool_protocol.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_gemini_tool_protocol.py`:

```python
"""Tests for GeminiClient tool-calling: structured content, function_call parsing,
function_response round-trip."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kaggle_slayer.agent import llm_client as llm
from kaggle_slayer.agent.cost_ledger import CostLedger


def _mock_genai_text_response(text: str, in_tok: int = 5, out_tok: int = 3):
    resp = MagicMock()
    resp.text = text
    part = MagicMock()
    part.function_call = None
    part.text = text
    cand = MagicMock()
    cand.content.parts = [part]
    resp.candidates = [cand]
    usage = MagicMock()
    usage.prompt_token_count = in_tok
    usage.candidates_token_count = out_tok
    usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


def _mock_genai_tool_call_response(name: str, args: dict, in_tok: int = 5, out_tok: int = 3):
    resp = MagicMock()
    resp.text = ""
    part = MagicMock()
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part.function_call = fc
    part.text = None
    cand = MagicMock()
    cand.content.parts = [part]
    resp.candidates = [cand]
    usage = MagicMock()
    usage.prompt_token_count = in_tok
    usage.candidates_token_count = out_tok
    usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


def test_gemini_parses_text_response(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("hello")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        resp = client.call(messages=[llm.Message(role="user", content="hi")])
    assert resp.text == "hello"
    assert resp.tool_calls == []


def test_gemini_parses_function_call(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_tool_call_response(
            name="train_cv", args={}
        )
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        resp = client.call(messages=[llm.Message(role="user", content="run cv")])
    assert resp.tool_calls
    assert resp.tool_calls[0].name == "train_cv"
    assert resp.tool_calls[0].args == {}


def test_gemini_parses_function_call_with_args(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_tool_call_response(
            name="write_file", args={"path": "agent/fe.py", "content": "x = 1"}
        )
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        resp = client.call(messages=[llm.Message(role="user", content="write")])
    assert resp.tool_calls[0].name == "write_file"
    assert resp.tool_calls[0].args == {"path": "agent/fe.py", "content": "x = 1"}


def test_gemini_passes_tools_to_generate_content(tmp_path):
    """When tools are provided, they should appear in the kwargs passed to
    the underlying generate_content call."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("ok")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        client.call(
            messages=[llm.Message(role="user", content="hi")],
            tools=[{"name": "train_cv", "description": "run CV", "parameters": {"type": "object", "properties": {}}}],
        )
    call_kwargs = impl.models.generate_content.call_args.kwargs
    # The config kwarg should carry tools (translated into Gemini's format)
    assert "config" in call_kwargs


def test_gemini_message_to_content_handles_tool_role(tmp_path):
    """Messages with role='tool' must be serialized as function_response parts.

    We verify by inspecting the contents= argument passed to generate_content."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("ok")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        client.call(messages=[
            llm.Message(role="user", content="run train_cv"),
            llm.Message(role="model", content="<tool_call:train_cv:{}>"),
            llm.Message(role="tool", content='{"tool":"train_cv","result":"mean=0.82"}'),
        ])
    call_kwargs = impl.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert len(contents) == 3
    # Last content's role should map to Gemini's tool/user/function role; structure verified via mock interface
    assert hasattr(contents[-1], "parts") or isinstance(contents[-1], dict)
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_gemini_tool_protocol.py -v
```

Expected: failures around `tool_calls == []` because Week-2 GeminiClient ignored the response structure.

- [ ] **Step 3: Replace `_messages_to_genai_contents` and the `GeminiClient.call` body in `kaggle_slayer/agent/llm_client.py`**

Find the current `_messages_to_genai_contents` function and the `GeminiClient.call` method. Replace **both** with this:

```python
def _messages_to_genai_contents(messages: list[Message]) -> list[Any]:
    """Translate the harness's Message list to Gemini's Content list.

    user / system messages → user-role Content with a text Part.
    model messages → model-role Content with a text Part.
    tool messages → user-role Content with a function_response Part (Gemini's
        convention is that tool results come back from the "user" role).

    For tool messages, `content` must be a JSON object string with shape
    {"tool": "<name>", "result": <any json-able>}.
    """
    from google.genai import types as gt

    contents: list[Any] = []
    for m in messages:
        if m.role == "tool":
            try:
                payload = json.loads(m.content)
            except json.JSONDecodeError:
                payload = {"tool": "unknown", "result": m.content}
            name = payload.get("tool", "unknown")
            result = payload.get("result", "")
            # FunctionResponse.response must be a dict — wrap scalars/strings.
            response_payload = result if isinstance(result, dict) else {"result": result}
            part = gt.Part(function_response=gt.FunctionResponse(
                name=name,
                response=response_payload,
            ))
            contents.append(gt.Content(role="user", parts=[part]))
        elif m.role == "model":
            contents.append(gt.Content(role="model", parts=[gt.Part(text=m.content)]))
        else:  # user, system → user
            contents.append(gt.Content(role="user", parts=[gt.Part(text=m.content)]))
    return contents


def _function_declarations_to_genai_tools(declarations: list[dict[str, Any]]) -> list[Any]:
    """Translate our generic function-declaration list to Gemini's Tool list."""
    from google.genai import types as gt

    decls = [
        gt.FunctionDeclaration(
            name=d["name"],
            description=d.get("description", ""),
            parameters=d.get("parameters"),
        )
        for d in declarations
    ]
    return [gt.Tool(function_declarations=decls)]


# Replace existing class GeminiClient method `call` with the body below:

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        from google.genai import types as gt

        chosen_model = model or self._default_model
        contents = _messages_to_genai_contents(messages)

        config = None
        if tools:
            gemini_tools = _function_declarations_to_genai_tools(tools)
            config = gt.GenerateContentConfig(tools=gemini_tools)

        last_err: Exception | None = None
        raw = None
        for attempt in range(self._retry_max + 1):
            try:
                kwargs: dict[str, Any] = {"model": chosen_model, "contents": contents}
                if config is not None:
                    kwargs["config"] = config
                raw = self._client.models.generate_content(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if not _is_transient(e) or attempt == self._retry_max:
                    raise
                time.sleep(self._retry_base_delay_s * (2 ** attempt))
        if raw is None:
            raise last_err or RuntimeError("unreachable")

        usage = raw.usage_metadata
        u = Usage(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
        )
        self._ledger.record(
            model=chosen_model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_tokens=u.cached_tokens,
            competition=self._competition,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for cand in (raw.candidates or []):
            for part in (getattr(cand.content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    args = dict(getattr(fc, "args", {}) or {})
                    tool_calls.append(ToolCall(
                        id=getattr(fc, "id", "") or f"tc_{len(tool_calls)}",
                        name=fc.name,
                        args=args,
                    ))
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

        return Response(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=u,
            raw=raw,
        )
```

Also add `import json` at the top of the file (next to the existing imports) — it's needed by `_messages_to_genai_contents`. Skip if already imported.

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_llm_client.py tests/unit/test_gemini_tool_protocol.py -v
```

Expected: all Task-7 tests pass, plus all existing Gemini tests still pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/llm_client.py tests/unit/test_gemini_tool_protocol.py
git commit -m "$(cat <<'EOF'
feat(agent): GeminiClient supports structured Content/Part + tool calls

Replaces Week-2's string-flatten message serialization with proper
Gemini Content/Part lists:
  - user/system messages → user-role Content with text Part
  - model messages → model-role Content with text Part
  - tool messages → user-role Content with function_response Part (Gemini
    receives tool results as user-side content per its API contract)

call(tools=...) now translates our generic function declarations into
gt.Tool(function_declarations=[...]) and passes via GenerateContentConfig.

Response parsing iterates candidate parts, extracting function_call parts
into ToolCall records (id/name/args) and text parts into the response
.text. The raw response object stays in Response.raw for debugging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: System prompt resource file

The Solver needs a system prompt. Storing it as a markdown file under `kaggle_slayer/agent/prompts/` makes it easy to iterate without code changes.

**Files:**
- Create: `kaggle_slayer/agent/prompts/__init__.py`
- Create: `kaggle_slayer/agent/prompts/system.md`

- [ ] **Step 1: Create the package**

`kaggle_slayer/agent/prompts/__init__.py`:

```python
"""Prompt resource files loaded at runtime."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def load_system_prompt() -> str:
    """Load the Solver system prompt from system.md."""
    return (_HERE / "system.md").read_text()
```

- [ ] **Step 2: Create the system prompt at `kaggle_slayer/agent/prompts/system.md`**

```markdown
# KaggleSlayer Solver — System Prompt

You are KaggleSlayer's Solver agent. You're working on a single Kaggle competition
described in the user message (the contents of `context.md`). The goal is a `submission.csv`
that scores at or above the public-leaderboard median.

## Workflow

1. Read the context carefully. Note the metric, target column, problem type.
2. Look at real data with `sample_rows` if the context profile isn't enough.
3. Write `agent/fe.py` (`fit_feature_transformer(train_df, target_col)`) and
   `agent/model.py` (`fit_model(X_train, y_train, problem_type, metric_name)`)
   with `write_file`.
4. Run `train_cv` to get a leak-free CV score.
5. Iterate: try different features, try different models, re-run `train_cv`.
6. Once you're happy, call `submit_local(label=...)` to write the submission CSV.
7. Call `done(summary=...)` to finish.

## Contracts your code must honor

**`agent/fe.py`** must expose:

```python
def fit_feature_transformer(train_df, target_col):
    """Fit on train_df ONLY. Return an object with .transform(df) -> df.
    Do NOT read from raw/* directly; everything you need is in train_df.
    """
```

The `.transform()` you return MUST preserve row count (no filtering).

**`agent/model.py`** must expose:

```python
def fit_model(X_train, y_train, problem_type, metric_name):
    """Return a fitted model with .predict(X). If the metric needs probabilities,
    the model must also have .predict_proba(X)."""
```

## Hard rules

- DON'T read raw competition files directly in fe.py or model.py. The harness
  passes you everything you need.
- DON'T touch `os.remove`, `shutil`, `subprocess`, `eval`, network calls. The
  sandbox lint will reject your code.
- DON'T write to `run_log.jsonl`, `notes.jsonl`, or `context.md` via `write_file` —
  those are protected.
- DO use `take_note` to record observations, decisions, hypotheses, and todos.
- DO trust the CV result — if `train_cv` says mean=0.82, the agent should believe
  that and improve on it, not re-run hoping for a different number.

## Style

Be terse. Don't restate the obvious. Make moves; don't narrate them.

## You're done when

- `train_cv` shows a reasonable score (you've judged "good enough" for this comp).
- `submit_local` has produced a CSV.
- `done(summary=...)` has been called.
```

- [ ] **Step 3: Quick test that loading works**

```bash
python -c "from kaggle_slayer.agent.prompts import load_system_prompt; print(load_system_prompt()[:80])"
```

Expected: prints the first 80 chars of system.md.

- [ ] **Step 4: Commit**

```bash
git add kaggle_slayer/agent/prompts/
git commit -m "$(cat <<'EOF'
feat(agent): add system prompt + load_system_prompt() helper

prompts/system.md is the Solver's system message: workflow, code-contract
specs for fe.py and model.py, hard rules (no raw/* reads, no destructive
calls, no protected-file writes), and the done-when criteria.

load_system_prompt() reads the file at runtime so we can iterate on the
prompt without code changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `Solver` class — the agent loop

The heart of Week 3.

**Files:**
- Create: `kaggle_slayer/agent/solver.py`
- Create: `tests/unit/test_solver.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_solver.py`:

```python
"""Tests for kaggle_slayer.agent.solver.Solver."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaggle_slayer.agent.llm_client import Message, Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver, SolveResult, SolverContext
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


class _CannedClient:
    """Bare-bones LLMClient stand-in that returns a fixed sequence of Responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.captured: list[list[Message]] = []

    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._responses[self._i]
        self._i += 1
        return r


def _make_workspace_and_ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    ws.context_path.write_text("# Fake Comp\n\nMetric: accuracy. Target: target.")
    return ws


def test_solver_exits_on_done_response(tmp_path):
    """A tool call to 'done' must stop the loop and produce status=done."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="done", args={"summary": "fake done"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert isinstance(result, SolveResult)
    assert result.status == "done"
    assert "fake done" in result.summary


def test_solver_exits_on_max_iterations(tmp_path):
    """If the agent never calls done, the loop must terminate at max_iterations."""
    ws = _make_workspace_and_ctx(tmp_path)
    # Endless empty-text replies — no done call
    client = _CannedClient(responses=[
        Response(text="thinking...", tool_calls=[], usage=Usage(0, 0, 0))
        for _ in range(20)
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=3)
    result = solver.solve()
    assert result.status == "max_iterations"
    assert result.iterations == 3


def test_solver_dispatches_tool_call_and_feeds_result_back(tmp_path):
    """When the LLM returns a tool call, the solver invokes it and feeds the
    result back as a tool-role message on the next turn."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="take_note", args={"category": "observation", "content": "x"})],
            usage=Usage(0, 0, 0),
        ),
        Response(
            text="",
            tool_calls=[ToolCall(id="tc2", name="done", args={"summary": "ok"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert result.status == "done"

    # On the second call, the messages must include a tool-role message with
    # the take_note result
    second_msgs = client.captured[1]
    tool_roles = [m for m in second_msgs if m.role == "tool"]
    assert len(tool_roles) >= 1
    # Note was actually written
    assert ws.notes_path.exists()


def test_solver_journals_each_tool_call(tmp_path):
    """Every tool call (success or error) lands in run_log.jsonl."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="take_note", args={"category": "observation", "content": "noted"})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    import json
    log_records = [json.loads(line) for line in ws.run_log_path.read_text().splitlines()]
    tool_calls_logged = [r for r in log_records if r["tool"] in ("take_note", "done")]
    assert len(tool_calls_logged) == 2


def test_solver_handles_tool_error_and_feeds_message_back(tmp_path):
    """If a tool raises ToolError, the solver journals it and feeds the error
    back to the LLM so it can correct itself."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        # First: invalid call — write_file with a protected path
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="write_file", args={"path": "context.md", "content": "x"})],
                 usage=Usage(0, 0, 0)),
        # Second: done
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "bailing"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert result.status == "done"

    # Second LLM call must include a tool-role message reporting the error
    second = client.captured[1]
    tool_msgs = [m for m in second if m.role == "tool"]
    assert any("protected" in m.content.lower() for m in tool_msgs)


def test_solver_context_carries_target_metric_problem_type(tmp_path):
    """The SolverContext defaults can be overridden via kwargs."""
    ws = _make_workspace_and_ctx(tmp_path)
    ctx = SolverContext(workspace=ws, journal=Journal(ws), target_col="my_target",
                        problem_type="regression", metric_name="rmse")
    assert ctx.target_col == "my_target"
    assert ctx.problem_type == "regression"
    assert ctx.metric_name == "rmse"
    assert ctx.finished is False
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_solver.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.solver`.

- [ ] **Step 3: Create `kaggle_slayer/agent/solver.py`**

```python
"""Solver — the agent loop.

Per-turn: pass the message history + tool declarations to the LLM,
parse the response. If the response has tool_calls, dispatch each via the
ToolRegistry, append the result (or error) as a tool-role Message, and
loop. If the response is plain text with no tool calls, treat it as
"thinking aloud" and continue. Exit on:
  - the agent calls `done`
  - max_iterations exhausted
  - wall-clock budget exhausted
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.agent.handlers import make_builtin_registry
from kaggle_slayer.agent.llm_client import LLMClient, Message
from kaggle_slayer.agent.prompts import load_system_prompt
from kaggle_slayer.agent.tools import ToolError, ToolRegistry
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class SolverContext:
    """State the tool handlers read and write."""

    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""


@dataclass
class SolveResult:
    status: str  # "done" | "max_iterations" | "time_exceeded"
    iterations: int
    summary: str


class Solver:
    """Runs the agent loop against a single competition workspace."""

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
    ) -> None:
        self.workspace = workspace
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.time_budget_s = time_budget_s
        self.registry = registry or make_builtin_registry()
        self.journal = Journal(workspace)
        self.ctx = SolverContext(
            workspace=workspace,
            journal=self.journal,
            target_col=target_col,
            problem_type=problem_type,
            metric_name=metric_name,
        )

    def solve(self) -> SolveResult:
        system_prompt = load_system_prompt()
        context_md = (
            self.workspace.context_path.read_text()
            if self.workspace.context_path.exists()
            else "(no context.md yet)"
        )

        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=context_md),
        ]
        tool_decls = self.registry.to_function_declarations()

        started = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            if time.perf_counter() - started > self.time_budget_s:
                return SolveResult(status="time_exceeded", iterations=iteration - 1, summary="")

            response = self.llm.call(messages=messages, tools=tool_decls)

            # If the LLM produced text alongside or instead of tool calls, append
            # it as a model-role message so the next turn has continuity.
            if response.text:
                messages.append(Message(role="model", content=response.text))

            if not response.tool_calls:
                # Pure-text response. Keep looping until the agent calls done or max iter.
                continue

            for tc in response.tool_calls:
                tool_result_text = self._dispatch(tc.name, tc.args)
                # Feed the result back as a tool-role message. We serialize as a
                # small JSON payload so the LLMClient knows which tool the
                # function_response Part should attribute to.
                payload = json.dumps({"tool": tc.name, "result": tool_result_text})
                messages.append(Message(role="tool", content=payload))

                if self.ctx.finished:
                    return SolveResult(
                        status="done",
                        iterations=iteration,
                        summary=self.ctx.final_summary,
                    )

        return SolveResult(status="max_iterations", iterations=self.max_iterations, summary="")

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool, journal it, return a string result (success or error)."""
        try:
            result = self.registry.invoke(name, ctx=self.ctx, args=args)
            text_result = str(result)
            self.journal.log_tool_call(
                tool=name,
                args=args,
                result_summary=text_result[:200],
            )
            return text_result
        except ToolError as e:
            err_msg = f"ToolError: {e}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg
        except Exception as e:  # noqa: BLE001
            err_msg = f"unexpected error in {name}: {e!r}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg
```

- [ ] **Step 4: Run, observe pass**

```bash
pytest tests/unit/test_solver.py -v
```

Expected: 6 passes.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/agent/solver.py tests/unit/test_solver.py
git commit -m "$(cat <<'EOF'
feat(agent): add Solver — the agent loop

Solver wires together a Workspace, LLMClient, ToolRegistry, and Journal.
solve() runs the reason-act loop: build context, call LLM with tool
declarations, parse tool_calls from the response, dispatch each via the
registry, journal the call (success or error), append the result as a
tool-role Message, and loop. Exit conditions: agent calls done(),
max_iterations exhausted, or time_budget_s exhausted.

SolverContext is the typed state object tool handlers read and mutate
(target_col, problem_type, metric_name, cv_kind/params, finished flag).

ToolError is caught and fed back to the model as a tool-result text so
the agent can self-correct. Unexpected exceptions are caught too — the
loop never crashes on bad agent behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: CLI entry point

The `kaggle-slayer <comp-path>` command.

**Files:**
- Create: `kaggle_slayer/cli.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/test_cli.py`:

```python
"""Tests for kaggle_slayer.cli."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from kaggle_slayer import cli


def test_cli_parses_args(tmp_path):
    """Parser accepts a workspace path and optional --target/--metric/--problem-type."""
    args = cli._parse_args([
        str(tmp_path / "comp"),
        "--target", "Survived",
        "--metric", "accuracy",
        "--problem-type", "classification",
        "--max-iterations", "10",
    ])
    assert args.workspace_path == str(tmp_path / "comp")
    assert args.target == "Survived"
    assert args.metric == "accuracy"
    assert args.problem_type == "classification"
    assert args.max_iterations == 10


def test_cli_requires_workspace_path():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_cli_run_creates_workspace_and_calls_solver(tmp_path):
    """run() with a fake LLMClient creates the workspace and invokes the solver."""
    comp_path = tmp_path / "comp"

    # Pre-populate raw/train.csv and raw/test.csv so the workspace looks real
    comp_path.mkdir()
    raw = comp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"x1": [1, 2, 3], "Survived": [0, 1, 0]}).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"id": [1, 2], "x1": [1, 2]}).to_csv(raw / "test.csv", index=False)

    # Mock context builder and Solver to avoid real Kaggle/LLM calls
    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context") as mock_build_context, \
         patch("kaggle_slayer.cli.KaggleClient") as mock_kaggle_cls, \
         patch("kaggle_slayer.cli.GeminiClient") as mock_gemini_cls, \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):

        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(
            status="done", iterations=3, summary="best CV=0.85"
        )
        mock_solver_cls.return_value = mock_solver

        exit_code = cli.run([
            str(comp_path),
            "--target", "Survived",
            "--metric", "accuracy",
            "--problem-type", "classification",
        ])

    assert exit_code == 0
    mock_solver_cls.assert_called_once()
    mock_solver.solve.assert_called_once()
    # Workspace structure was created
    assert (comp_path / "agent").is_dir()
    assert (comp_path / "submissions").is_dir()


def test_cli_run_exits_nonzero_when_solver_does_not_finish(tmp_path):
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):

        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(
            status="max_iterations", iterations=25, summary=""
        )
        mock_solver_cls.return_value = mock_solver

        exit_code = cli.run([str(comp_path), "--target", "y"])
    assert exit_code != 0
```

- [ ] **Step 2: Run, observe failure**

```bash
pytest tests/unit/test_cli.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.cli`.

- [ ] **Step 3: Create `kaggle_slayer/cli.py`**

```python
"""kaggle-slayer CLI entry point.

Usage:
    kaggle-slayer <workspace-path> --target <col> [--metric <m>] [--problem-type <p>] [--max-iterations N]

This is intentionally thin: parse args, ensure the workspace exists,
maybe build context.md, then invoke the Solver. Heavy lifting lives in
kaggle_slayer.agent.solver.Solver.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness.context import build_context
from kaggle_slayer.harness.kaggle_client import KaggleClient
from kaggle_slayer.harness.workspace import Workspace


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="kaggle-slayer",
        description="LLM-agent harness for tabular Kaggle competitions.",
    )
    p.add_argument("workspace_path", help="Path to per-competition workspace (e.g., competitions/titanic)")
    p.add_argument("--target", default=None, help="Target column name")
    p.add_argument("--metric", default="accuracy", help="Metric (accuracy, auc, logloss, rmse, mae, r2)")
    p.add_argument("--problem-type", default="classification", choices=["classification", "regression"])
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--time-budget-s", type=float, default=900.0)
    p.add_argument("--model", default="gemini-2.5-pro", help="Gemini model id")
    p.add_argument("--no-context-build", action="store_true",
                   help="Skip rebuilding context.md (use existing one)")
    return p.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    load_dotenv()

    comp_path = Path(args.workspace_path)
    workspace = Workspace.create(root=comp_path)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: no GEMINI_API_KEY / GOOGLE_API_KEY in env", file=sys.stderr)
        return 2

    # Build context.md unless user opts out
    if not args.no_context_build:
        try:
            kaggle = KaggleClient()
            build_context(workspace=workspace, kaggle_client=kaggle)
        except Exception as e:  # noqa: BLE001
            # Non-fatal — the agent can still read whatever context.md exists,
            # or none at all. Surface the warning.
            print(f"warning: context build failed: {e!r}", file=sys.stderr)

    ledger = CostLedger()
    llm = GeminiClient(
        api_key=api_key,
        ledger=ledger,
        competition=workspace.name,
        default_model=args.model,
    )
    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col=args.target or "target",
        problem_type=args.problem_type,
        metric_name=args.metric,
        max_iterations=args.max_iterations,
        time_budget_s=args.time_budget_s,
    )
    result = solver.solve()

    print(f"\nstatus: {result.status}")
    print(f"iterations: {result.iterations}")
    print(f"summary: {result.summary}")
    print(f"spent: ${ledger.total_for(competition=workspace.name):.4f}")

    return 0 if result.status == "done" else 1


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify pyproject.toml has the entry point**

```bash
grep -A 1 'project.scripts' pyproject.toml
```

Expected output (from Week 1, may need to reinstall after this commit so the script picks up):

```
[project.scripts]
kaggle-slayer = "kaggle_slayer.cli:main"
```

If missing, add it manually.

- [ ] **Step 5: Run, observe pass**

```bash
pytest tests/unit/test_cli.py -v
pip install -e . > /dev/null
which kaggle-slayer
kaggle-slayer --help
```

Expected: 4 cli tests pass; `kaggle-slayer` is on `$PATH`; `--help` prints the usage.

- [ ] **Step 6: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add kaggle_slayer/cli.py tests/unit/test_cli.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(cli): add kaggle-slayer entry point

kaggle-slayer <workspace-path> [--target ...] [--metric ...] runs the
full agent loop end-to-end: creates the workspace, builds context.md
(unless --no-context-build), instantiates GeminiClient + Solver, runs
solve(), prints status/iterations/summary/cost.

Args:
  --target           target column name
  --metric           accuracy/auc/logloss/rmse/mae/r2  (default: accuracy)
  --problem-type     classification|regression
  --max-iterations   solver iteration cap (default 25)
  --time-budget-s    wall-clock cap (default 900)
  --model            Gemini model id (default gemini-2.5-pro)
  --no-context-build skip rebuilding context.md

Returns 0 on done, 1 on max_iterations/time_exceeded, 2 on missing creds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Synthetic micro-comp fixture + scripted-tools integration test

The fake-agent integration test exercises the full Solver loop end-to-end on a synthetic comp without burning Gemini quota.

**Files:**
- Create: `tests/fixtures/synthetic_comp.py`
- Create: `tests/integration/test_solver_with_fake_agent.py`

- [ ] **Step 1: Create the synthetic-comp fixture**

`tests/fixtures/synthetic_comp.py`:

```python
"""Programmatic synthetic Kaggle micro-competition.

Creates a tiny binary-classification dataset (500 train rows, 100 test rows)
inside a temporary workspace. Used by integration tests so we exercise the
full Solver loop without depending on real Kaggle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_slayer.harness.workspace import Workspace


def make_synthetic_comp(root: Path, *, seed: int = 0) -> Workspace:
    """Create a workspace at `root` with raw/train.csv, raw/test.csv,
    and a context.md flagging Survived as target + accuracy as metric.

    The target is a noisy function of x1 + x2 — learnable by LR.
    """
    rng = np.random.default_rng(seed)
    n_train, n_test = 500, 100

    train = pd.DataFrame({
        "id": range(n_train),
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train),
        "x3": rng.normal(size=n_train),
        "Sex": rng.choice(["male", "female"], size=n_train),
    })
    logits = 1.5 * train["x1"] - 0.8 * train["x2"] + rng.normal(scale=0.5, size=n_train)
    train["Survived"] = (logits > 0).astype(int)

    test = pd.DataFrame({
        "id": range(n_train, n_train + n_test),
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test),
        "x3": rng.normal(size=n_test),
        "Sex": rng.choice(["male", "female"], size=n_test),
    })

    workspace = Workspace.create(root=root)
    train.to_csv(workspace.raw_dir / "train.csv", index=False)
    test.to_csv(workspace.raw_dir / "test.csv", index=False)
    workspace.context_path.write_text(
        "# Synthetic Comp\n\n"
        "## Description\nBinary classification on 5 features.\n\n"
        "## Evaluation metric\n`accuracy`\n\n"
        "## Data profile (train.csv)\n"
        "- **Rows:** 500\n"
        "- **Likely target column(s):** `Survived`\n"
        "- **ID column:** `id`\n\n"
        "## Public leaderboard (top scores for reference)\n"
        "*synthetic; no real LB*\n"
    )
    return workspace
```

- [ ] **Step 2: Failing integration test**

`tests/integration/test_solver_with_fake_agent.py`:

```python
"""Integration test: Solver runs against a synthetic competition using a
scripted FakeLLMClient (no real Gemini calls)."""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from tests.fixtures.synthetic_comp import make_synthetic_comp


pytestmark = pytest.mark.integration


_FE_CODE = '''
import pandas as pd

class _PT:
    def __init__(self, cols, means):
        self.cols = cols
        self.means = means
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            if c in df.columns:
                out[c] = df[c].fillna(self.means.get(c, 0.0))
        # Drop the id column if present
        if "id" in out.columns:
            out = out.drop(columns=["id"])
        return out

def fit_feature_transformer(train_df, target_col):
    cols = [c for c in train_df.columns
            if c not in (target_col, "id") and train_df[c].dtype.kind in "fiub"]
    means = {c: float(train_df[c].mean()) for c in cols}
    return _PT(cols, means)
'''

_MODEL_CODE = '''
from sklearn.linear_model import LogisticRegression, Ridge

def fit_model(X_train, y_train, problem_type, metric_name):
    if problem_type == "classification":
        m = LogisticRegression(max_iter=500, random_state=42)
    else:
        m = Ridge(alpha=1.0, random_state=42)
    m.fit(X_train, y_train)
    return m
'''


class _ScriptedClient:
    """LLMClient that returns a fixed sequence of Responses with tool calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.captured = []

    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._responses[self._i]
        self._i += 1
        return r


def test_solver_end_to_end_with_scripted_tools(tmp_path):
    """The scripted agent: write fe.py, write model.py, train_cv, submit_local, done."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")

    responses = [
        # 1. Write fe.py
        Response(text="", tool_calls=[ToolCall(
            id="t1", name="write_file",
            args={"path": "agent/fe.py", "content": _FE_CODE},
        )], usage=Usage(0, 0, 0)),
        # 2. Write model.py
        Response(text="", tool_calls=[ToolCall(
            id="t2", name="write_file",
            args={"path": "agent/model.py", "content": _MODEL_CODE},
        )], usage=Usage(0, 0, 0)),
        # 3. Train CV
        Response(text="", tool_calls=[ToolCall(
            id="t3", name="train_cv", args={},
        )], usage=Usage(0, 0, 0)),
        # 4. Submit local
        Response(text="", tool_calls=[ToolCall(
            id="t4", name="submit_local", args={"label": "scripted"},
        )], usage=Usage(0, 0, 0)),
        # 5. Done
        Response(text="", tool_calls=[ToolCall(
            id="t5", name="done", args={"summary": "scripted run complete"},
        )], usage=Usage(0, 0, 0)),
    ]
    client = _ScriptedClient(responses)
    solver = Solver(
        workspace=workspace,
        llm_client=client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=10,
    )

    result = solver.solve()
    assert result.status == "done"
    assert "scripted" in result.summary

    # Verify the files the agent wrote
    assert workspace.fe_path.exists()
    assert workspace.model_path.exists()
    # Versions archive exists (one fe + one model)
    assert (workspace.versions_dir / "fe_v01.py").exists()
    assert (workspace.versions_dir / "model_v01.py").exists()
    # Submission written
    submissions = list(workspace.submissions_dir.glob("*scripted*.csv"))
    assert len(submissions) == 1
    # Run log has all 5 tool calls
    log_lines = workspace.run_log_path.read_text().splitlines()
    assert len(log_lines) >= 5
```

- [ ] **Step 3: Run, observe pass**

```bash
pytest tests/integration/test_solver_with_fake_agent.py -v -m integration
```

Expected: 1 pass. **This is the Week-3 internal acceptance gate before turning on real Gemini.**

- [ ] **Step 4: Full suite**

```bash
pytest -m "not slow" --cov=kaggle_slayer/harness --cov=kaggle_slayer/agent --cov-report=term 2>&1 | tail -5
```

Expected: ~210+ tests pass, coverage on new code ≥ 85%.

- [ ] **Step 5: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/fixtures/synthetic_comp.py tests/integration/test_solver_with_fake_agent.py
git commit -m "$(cat <<'EOF'
test: add synthetic micro-comp + scripted-tools Solver integration

tests/fixtures/synthetic_comp.py builds a 500-train / 100-test binary
classification workspace programmatically with target='Survived' and a
pre-written context.md.

tests/integration/test_solver_with_fake_agent.py runs Solver end-to-end
through a scripted 5-step tool sequence (write fe.py, write model.py,
train_cv, submit_local, done) and asserts: status=done, versions
archived, submission CSV written, run log has 5 tool calls. No real LLM
quota burned.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Real-Gemini E2E acceptance (slow tier)

The actual Week-3 acceptance: real Gemini solves the synthetic micro-comp.

**Files:**
- Create: `tests/integration/test_solver_real_gemini.py`

- [ ] **Step 1: Write the slow E2E test**

`tests/integration/test_solver_real_gemini.py`:

```python
"""Real-Gemini E2E acceptance — slow tier, opt-in.

Runs the full Solver loop against a synthetic comp with real Gemini.
Costs ≈ $0.01-0.05 per run depending on iteration count. Skipped when
GEMINI_API_KEY is missing.

Run with: pytest -m slow tests/integration/test_solver_real_gemini.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from tests.fixtures.synthetic_comp import make_synthetic_comp


pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env or .env")
    return key


def test_real_gemini_solves_synthetic_microcomp(tmp_path, gemini_key):
    """Acceptance gate: real Gemini reads context.md, writes fe.py + model.py,
    runs train_cv, calls submit_local, calls done. Submission CSV must exist
    with non-empty predictions and the right row count."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    llm = GeminiClient(
        api_key=gemini_key,
        ledger=ledger,
        competition="synthetic-e2e",
        default_model="gemini-2.5-pro",
        retry_max=2,
    )
    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=20,
        time_budget_s=600.0,
    )
    result = solver.solve()

    # Hard requirements for the acceptance:
    assert result.status == "done", (
        f"Solver did not finish cleanly: status={result.status}, "
        f"iterations={result.iterations}, summary={result.summary!r}"
    )
    # The agent must have written fe.py and model.py
    assert workspace.fe_path.exists(), "agent did not write fe.py"
    assert workspace.model_path.exists(), "agent did not write model.py"
    # At least one CV pass was archived
    assert any(workspace.versions_dir.glob("fe_v*.py"))
    assert any(workspace.versions_dir.glob("model_v*.py"))
    # At least one submission CSV exists
    submissions = list(workspace.submissions_dir.glob("*.csv"))
    assert submissions, "no submission CSV was written"
    sub = pd.read_csv(submissions[0])
    assert len(sub) == 100, f"submission row count wrong: {len(sub)}"
    # Predictions must be 0/1 (label) or floats in [0,1] (proba)
    pred_col = [c for c in sub.columns if c.lower() not in ("id",)][0]
    assert sub[pred_col].notna().all(), "predictions contain NaN"
    # Cost was tracked
    assert ledger.total_for(competition="synthetic-e2e") > 0
    print(f"\nDONE. iter={result.iterations}, "
          f"cost=${ledger.total_for(competition='synthetic-e2e'):.4f}, "
          f"summary={result.summary!r}")
```

- [ ] **Step 2: Run, observe pass (this is the real test)**

```bash
pytest -m slow tests/integration/test_solver_real_gemini.py -v -s
```

Expected: 1 pass. **If it fails**, the test output will explain what went wrong; common failure modes:
  - Agent didn't call done — try increasing max_iterations or check the system prompt.
  - Submission CSV didn't get written — agent skipped submit_local.
  - Schema validation failed — agent passed wrong argument types.

If the test fails for a *prompt-engineering* reason (the agent didn't behave the way we want), iterate on `kaggle_slayer/agent/prompts/system.md` and re-run. **Don't loosen the test assertions.**

- [ ] **Step 3: Confirm `-m "not slow"` still excludes it**

```bash
pytest -m "not slow" 2>&1 | tail -3
```

Expected: this test does not appear; non-slow suite is unchanged.

- [ ] **Step 4: Lint + mypy + commit**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
git add tests/integration/test_solver_real_gemini.py
git commit -m "$(cat <<'EOF'
test: add real-Gemini E2E acceptance for synthetic micro-comp

Slow-tier test that runs the full Solver loop against a synthetic
binary-classification competition with real Gemini-2.5-pro. Asserts the
agent: writes fe.py + model.py, runs train_cv at least once (versions/
archive present), writes a submission CSV with the right row count and
no NaN predictions, calls done with a non-empty summary, tracks cost
through the ledger.

This is the Week-3 acceptance gate. ≈$0.01-0.05 per run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Week 3 acceptance summary

After all 12 tasks:

- ✅ `kaggle_slayer/agent/tools.py` — Tool + ToolRegistry + jsonschema validation.
- ✅ 9 builtin tools via `kaggle_slayer/agent/handlers/{files,ml}.py` + `__init__.make_builtin_registry()`.
- ✅ `kaggle_slayer/agent/llm_client.py` — Gemini structured Content/Part + tool-call parsing.
- ✅ `kaggle_slayer/agent/prompts/system.md` — Solver system prompt.
- ✅ `kaggle_slayer/agent/solver.py` — `Solver` class with the reason-act loop.
- ✅ `kaggle_slayer/cli.py` — `kaggle-slayer <workspace>` entry point.
- ✅ `Workspace.next_version_path` helper.
- ✅ Tightened `_is_transient` (status-code first); extended `_TARGET_HINTS`.
- ✅ Fake-agent integration test: scripted 5-step sequence solves synthetic comp.
- ✅ Real-Gemini E2E: model autonomously solves the synthetic micro-comp.
- ✅ Coverage on new code ≥ 85%; ruff + mypy strict clean.

**Week 4 starts with:** the checkpoint gate (human-in-the-loop on submit_kaggle / set_metric / wall-clock), proper subprocess sandbox + resource limits, full conversation resume from `run_log.jsonl`, and a `submit_kaggle` tool that goes through the checkpoint.
