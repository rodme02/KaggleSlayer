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
