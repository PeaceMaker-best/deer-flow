"""Recovery-generated identifiers must coexist with provider-owned identifiers."""

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware


@pytest.mark.parametrize("surface", ["structured", "invalid", "raw"])
@pytest.mark.parametrize("completed", [False, True])
def test_generated_id_does_not_collide_with_a_valid_sibling(surface, completed):
    source = {"structured": "call", "invalid": "invalid", "raw": "raw"}[surface]
    reserved = f"deerflow_synthetic_tool_call_0_{source}_1"
    calls = [{"name": "read_file", "id": reserved, "args": {}}, {"name": "bash", "id": "", "args": {}}]
    if surface == "structured":
        message = AIMessage(content="", tool_calls=calls)
    elif surface == "invalid":
        message = AIMessage(content="", invalid_tool_calls=[{**call, "args": "{}", "error": "historical invalid call"} for call in calls])
    else:
        raw = [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": "{}"}} for call in calls]
        message = AIMessage.model_construct(content="", tool_calls=[], invalid_tool_calls=[], additional_kwargs={"tool_calls": raw})
    messages = [message, ToolMessage(content="first result", tool_call_id=reserved, name="read_file")]
    if completed:
        messages.append(ToolMessage(content="second result", tool_call_id="", name="bash"))
    before = deepcopy([m.model_dump() for m in messages])
    mw = DanglingToolCallMiddleware()
    projected = mw._build_patched_messages(messages) or messages
    payload = _convert_message_to_dict(projected[0])
    ids = [call["id"] for call in payload["tool_calls"]]
    assert len(set(ids)) == 2
    assert ids[0] == reserved
    assert [result.tool_call_id for result in projected[1:]] == ids
    assert projected[1].content == "first result"
    if completed:
        assert projected[2].content == "second result"
    else:
        assert projected[2].status == "error"
    assert [m.model_dump() for m in messages] == before
    assert mw._build_patched_messages(projected) is None


def test_recovery_skips_multiple_reserved_candidates_deterministically():
    root = "deerflow_synthetic_tool_call_0_call_2"
    message = AIMessage(
        content="",
        tool_calls=[
            {"id": root, "name": "read_file", "args": {}},
            {"id": root + "_1", "name": "grep", "args": {}},
            {"id": "", "name": "bash", "args": {}},
        ],
    )
    mw = DanglingToolCallMiddleware()
    first = mw._build_patched_messages([message])
    second = mw._build_patched_messages([message])
    first_ids = [x["id"] for x in _convert_message_to_dict(first[0])["tool_calls"]]
    assert len(set(first_ids)) == 3
    assert first_ids[:2] == [root, root + "_1"]
    assert first == second


@pytest.mark.parametrize("other_view", [False, True])
def test_recovery_reserves_ids_before_visiting_later_calls(other_view):
    reserved = "deerflow_synthetic_tool_call_0_call_0"
    missing = {"id": "", "name": "bash", "args": {}}
    sibling = {"id": reserved, "name": "read_file", "args": {}}
    message = AIMessage(
        content="",
        tool_calls=[missing] if other_view else [missing, sibling],
        invalid_tool_calls=[{**sibling, "args": "{}", "error": "historical failure"}] if other_view else [],
    )
    projected = DanglingToolCallMiddleware()._build_patched_messages([message])
    ids = [call["id"] for call in _convert_message_to_dict(projected[0])["tool_calls"]]
    assert len(set(ids)) == 2
    assert ids[1] == reserved
    assert [result.tool_call_id for result in projected[1:]] == ids


def test_valid_reused_ids_in_separate_turns_remain_a_noop():
    messages = []
    for value in ("first", "second"):
        messages.extend(
            [
                AIMessage(content="", tool_calls=[{"id": "reused", "name": "bash", "args": {"command": value}}]),
                ToolMessage(content=value, tool_call_id="reused", name="bash"),
            ]
        )
    assert DanglingToolCallMiddleware()._build_patched_messages(messages) is None


def test_generated_id_cannot_consume_a_later_turns_result():
    reserved = "deerflow_synthetic_tool_call_0_call_0"
    messages = [
        AIMessage(content="", tool_calls=[{"id": "", "name": "bash", "args": {}}]),
        AIMessage(content="", tool_calls=[{"id": reserved, "name": "read_file", "args": {}}]),
        ToolMessage(content="later result", tool_call_id=reserved, name="read_file"),
    ]
    projected = DanglingToolCallMiddleware()._build_patched_messages(messages)
    assert projected[1].status == "error"
    assert projected[1].tool_call_id != reserved
    assert projected[3] == messages[2]


def test_generated_id_does_not_adopt_an_orphan_result():
    reserved = "deerflow_synthetic_tool_call_0_call_0"
    messages = [
        AIMessage(content="", tool_calls=[{"id": "", "name": "bash", "args": {}}]),
        ToolMessage(content="orphan result", tool_call_id=reserved, name="bash"),
    ]
    projected = DanglingToolCallMiddleware()._build_patched_messages(messages)
    assert len(projected) == 2
    assert projected[1].status == "error"
    assert projected[1].tool_call_id != reserved
    assert projected[1].content != "orphan result"
