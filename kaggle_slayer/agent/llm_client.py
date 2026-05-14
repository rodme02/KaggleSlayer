"""Provider-agnostic LLMClient protocol + dataclasses.

The protocol is intentionally tiny: a single `call(messages, tools)` method
returning a Response. Concrete implementations live alongside (GeminiClient
in this same module; Claude/OpenAI clients later if needed).

ToolCall captures one function-call request from the model. Usage captures
token counts so the harness can compute cost via the CostLedger.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kaggle_slayer.agent.cost_ledger import CostLedger


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


class TransientLLMError(Exception):
    """Retryable error — rate limit, timeout, transient 5xx, etc."""


def _make_genai_client(api_key: str) -> Any:  # noqa: ANN401 — google-genai is loosely typed
    """Construct a google-genai Client. Factored out so tests can patch it."""
    from google import genai  # noqa: PLC0415

    return genai.Client(api_key=api_key)


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


class GeminiClient:
    """Concrete LLMClient for Google Gemini via google-genai."""

    def __init__(
        self,
        *,
        api_key: str,
        ledger: CostLedger,
        competition: str,
        default_model: str = "gemini-2.5-flash",
        retry_max: int = 3,
        retry_base_delay_s: float = 1.0,
    ) -> None:
        self._client = _make_genai_client(api_key)
        self._ledger = ledger
        self._competition = competition
        self._default_model = default_model
        self._retry_max = retry_max
        self._retry_base_delay_s = retry_base_delay_s

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
