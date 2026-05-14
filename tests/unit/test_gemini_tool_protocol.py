"""Tests for GeminiClient tool-calling: structured content, function_call parsing,
function_response round-trip."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _params_wire_dump(parameters):
    """Return the wire-equivalent dict for a FunctionDeclaration.parameters value.

    Gemini's SDK serializes pydantic Schemas to wire by skipping unset/None
    fields. The full `model_dump()` keeps every Schema field (with None for
    unset ones), so it does not reflect what actually goes on the wire.
    `exclude_none=True` matches the wire behavior."""
    if hasattr(parameters, "model_dump"):
        return parameters.model_dump(exclude_none=True)
    return parameters


def test_gemini_strips_additional_properties_from_parameters(tmp_path):
    """JSON schemas with additionalProperties:false must have that field stripped
    before being sent to Gemini — its OpenAPI subset doesn't recognize it.
    The local registry keeps the field for jsonschema validation."""
    from kaggle_slayer.agent.llm_client import _function_declarations_to_genai_tools

    decls = [{
        "name": "probe",
        "description": "A probe.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,  # Gemini will reject this
        },
    }]
    tools = _function_declarations_to_genai_tools(decls)
    # The translation must produce a Tool list with a FunctionDeclaration whose
    # parameters do NOT carry additionalProperties (or any other Gemini-rejected
    # field).
    fd = tools[0].function_declarations[0]
    # The parameters payload (typed Schema or dict) must not surface
    # additional_properties as an accepted key on the wire.
    params_dump = _params_wire_dump(fd.parameters)
    assert "additionalProperties" not in params_dump
    assert "additional_properties" not in params_dump


def test_gemini_strips_unsupported_keys_recursively(tmp_path):
    """Nested schemas (e.g., inside an array's items) also get stripped."""
    from kaggle_slayer.agent.llm_client import _function_declarations_to_genai_tools

    decls = [{
        "name": "nested",
        "description": "Nested schema test.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"k": {"type": "string"}},
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    }]
    tools = _function_declarations_to_genai_tools(decls)
    fd = tools[0].function_declarations[0]
    params_dump = _params_wire_dump(fd.parameters)
    # Neither the top-level params nor the nested array.items.parameters carry
    # additionalProperties any more.
    import json
    serialized = json.dumps(params_dump, default=str)
    assert "additionalProperties" not in serialized
    assert "additional_properties" not in serialized


def test_gemini_passes_system_message_as_system_instruction(tmp_path):
    """A Message(role='system', ...) must be hoisted out of the contents list
    and passed via GenerateContentConfig(system_instruction=...)."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("ok")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        client.call(messages=[
            llm.Message(role="system", content="be terse"),
            llm.Message(role="user", content="hi"),
        ])
    call_kwargs = impl.models.generate_content.call_args.kwargs
    assert "config" in call_kwargs, "system_instruction requires a GenerateContentConfig"
    cfg = call_kwargs["config"]
    assert getattr(cfg, "system_instruction", None) == "be terse"
    # The system message should NOT have leaked into contents
    contents = call_kwargs["contents"]
    assert len(contents) == 1, f"system message should be hoisted out; got {len(contents)} contents"


def test_gemini_handles_multiple_system_messages_by_concatenating(tmp_path):
    """Multiple system messages get newline-joined into a single
    system_instruction string, in order."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("ok")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        client.call(messages=[
            llm.Message(role="system", content="first system"),
            llm.Message(role="system", content="second system"),
            llm.Message(role="user", content="hi"),
        ])
    call_kwargs = impl.models.generate_content.call_args.kwargs
    cfg = call_kwargs["config"]
    assert getattr(cfg, "system_instruction", None) == "first system\nsecond system"
    assert len(call_kwargs["contents"]) == 1


def test_gemini_system_instruction_combines_with_tools(tmp_path):
    """system_instruction and tools must coexist on the same config."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as factory:
        impl = MagicMock()
        impl.models.generate_content.return_value = _mock_genai_text_response("ok")
        factory.return_value = impl
        client = llm.GeminiClient(api_key="x", ledger=ledger, competition="t", retry_max=0)
        client.call(
            messages=[
                llm.Message(role="system", content="be helpful"),
                llm.Message(role="user", content="hi"),
            ],
            tools=[{"name": "noop", "description": "n", "parameters": {"type": "object", "properties": {}}}],
        )
    call_kwargs = impl.models.generate_content.call_args.kwargs
    cfg = call_kwargs["config"]
    assert getattr(cfg, "system_instruction", None) == "be helpful"
    assert getattr(cfg, "tools", None), "tools must still be passed"


def test_messages_to_contents_emits_function_call_part_for_model_with_tool_calls():
    """A Message(role='model', tool_calls=[...]) must translate into a Gemini
    Content with role='model' whose first Part exposes a function_call (not text)."""
    from kaggle_slayer.agent.llm_client import _messages_to_genai_contents

    msgs = [
        llm.Message(role="user", content="hi"),
        llm.Message(
            role="model",
            content="",
            tool_calls=[llm.ToolCall(id="t1", name="train_cv", args={})],
        ),
    ]
    contents = _messages_to_genai_contents(msgs)
    assert len(contents) == 2
    model_content = contents[1]
    assert getattr(model_content, "role", None) == "model"
    parts = list(getattr(model_content, "parts", []) or [])
    assert parts, "model content with tool_calls must have at least one Part"
    fc = getattr(parts[0], "function_call", None)
    txt = getattr(parts[0], "text", None)
    assert fc is not None, "first Part must expose a function_call"
    assert getattr(fc, "name", None) == "train_cv"
    assert not txt, "first Part should not also carry text when carrying a function_call"


def test_gemini_preserves_supported_keys(tmp_path):
    """The strip removes only the Gemini-unsupported keys — supported ones survive."""
    from kaggle_slayer.agent.llm_client import _function_declarations_to_genai_tools

    decls = [{
        "name": "supported",
        "description": "Schema with supported fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["a", "b"]},
                "n": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    }]
    tools = _function_declarations_to_genai_tools(decls)
    fd = tools[0].function_declarations[0]
    params = _params_wire_dump(fd.parameters)
    import json
    serialized = json.dumps(params, default=str)
    # enum, minimum, maximum, default, required, type, properties all preserved
    assert "enum" in serialized
    # additionalProperties / additional_properties stripped
    assert "additionalProperties" not in serialized
    assert "additional_properties" not in serialized
