"""FakeLLMClient — scripted responses for integration tests.

Implements the LLMClient protocol without burning real API quota. Each
call() pops the next ScriptedResponse off the script and returns it.
The captured `calls` list lets tests assert what the harness sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.agent.llm_client import Message, Response, Usage


@dataclass
class ScriptedResponse:
    text: str
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0))


@dataclass
class CapturedCall:
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    model: str | None


class FakeLLMClient:
    def __init__(self, *, script: list[ScriptedResponse]) -> None:
        self._script = list(script)
        self.calls: list[CapturedCall] = []

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        self.calls.append(CapturedCall(messages=list(messages), tools=tools, model=model))
        if not self._script:
            raise RuntimeError("FakeLLMClient script exhausted")
        scripted = self._script.pop(0)
        return Response(text=scripted.text, tool_calls=[], usage=scripted.usage)
