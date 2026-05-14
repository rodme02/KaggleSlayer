"""Provider-agnostic LLMClient protocol + dataclasses.

The protocol is intentionally tiny: a single `call(messages, tools)` method
returning a Response. Concrete implementations live alongside (GeminiClient
in this same module; Claude/OpenAI clients later if needed).

ToolCall captures one function-call request from the model. Usage captures
token counts so the harness can compute cost via the CostLedger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "model" | "system" | "tool"
    content: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Response:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any = field(default=None, repr=False)  # provider-specific response object


@runtime_checkable
class LLMClient(Protocol):
    """A provider-agnostic LLM interface."""

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        ...
