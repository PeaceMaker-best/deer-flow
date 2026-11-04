"""Malformed historical arguments must not put non-JSON constants on the wire."""

import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware


def _strict_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


@pytest.mark.parametrize("arguments", ['{"limit":NaN}', '{"limit":Infinity}', '{"limit":-Infinity}', '{"nested":[{"limit":NaN}]}'])
@pytest.mark.parametrize("surface", ["invalid", "raw"])
def test_historical_non_json_constants_are_repaired_before_serialization(arguments, surface):
    if surface == "invalid":
        message = AIMessage(content="", invalid_tool_calls=[{"name": "read_file", "id": "call-a", "args": arguments, "error": "invalid arguments"}])
    else:
        # A persisted raw provider fallback can outlive its parsed call views.
        message = AIMessage.model_construct(content="", tool_calls=[], invalid_tool_calls=[], additional_kwargs={"tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "read_file", "arguments": arguments}}]})
    before = deepcopy(message.model_dump())
    messages = [message]
    projected = DanglingToolCallMiddleware()._build_patched_messages(messages) or messages
    call = _convert_message_to_dict(projected[0])["tool_calls"][0]
    decoded = json.loads(call["function"]["arguments"], parse_constant=_strict_constant)
    assert decoded == {}
    assert isinstance(projected[1], ToolMessage)
    assert projected[1].tool_call_id == call["id"]
    assert message.model_dump() == before


@pytest.mark.parametrize("arguments", ['{"label":"NaN / Infinity / -Infinity"}', '{"values":[1.25,-2e4,null,true]}', '{"nested":{"label":"无限大"}}'])
def test_valid_argument_strings_preserve_their_original_bytes(arguments):
    message = AIMessage(content="", invalid_tool_calls=[{"name": "read_file", "id": "call-a", "args": arguments, "error": "historical parse failure"}])
    messages = [message, ToolMessage(content="existing result", tool_call_id="call-a")]
    projected = DanglingToolCallMiddleware()._build_patched_messages(messages) or messages
    assert _convert_message_to_dict(projected[0])["tool_calls"][0]["function"]["arguments"] == arguments
    assert projected[1].content == "existing result"


@pytest.mark.parametrize("arguments", ['{"limit":NaN}', '{"limit":Infinity}', '{"limit":-Infinity}', '{"nested":[{"limit":NaN}]}'])
@pytest.mark.parametrize("completed", [False, True])
def test_constants_parsed_into_structured_calls_are_repaired(arguments, completed):
    # Use the normal constructor: LangChain itself accepts these constants and
    # promotes the raw provider call to a parsed dictionary.
    message = AIMessage(content="", additional_kwargs={"tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "read_file", "arguments": arguments}}]})
    assert message.tool_calls and not message.invalid_tool_calls
    before = json.dumps(message.model_dump(), sort_keys=True)
    messages = [message]
    if completed:
        messages.append(ToolMessage(content="existing result", tool_call_id="call-a"))
    projected = DanglingToolCallMiddleware()._build_patched_messages(messages) or messages
    call = _convert_message_to_dict(projected[0])["tool_calls"][0]
    assert json.loads(call["function"]["arguments"], parse_constant=_strict_constant) == {}
    assert projected[1].tool_call_id == call["id"]
    if completed:
        assert projected[1].content == "existing result"
    else:
        assert projected[1].status == "error"
    assert json.dumps(message.model_dump(), sort_keys=True) == before


def test_valid_structured_arguments_and_results_remain_a_noop():
    arguments = {"values": [1.25, -20000, None, True], "label": "NaN / Infinity / 无限大"}
    message = AIMessage(content="", tool_calls=[{"id": "call-a", "name": "read_file", "args": arguments}])
    messages = [message, ToolMessage(content="existing result", tool_call_id="call-a")]
    assert DanglingToolCallMiddleware()._build_patched_messages(messages) is None
