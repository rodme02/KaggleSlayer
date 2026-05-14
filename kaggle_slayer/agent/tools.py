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
