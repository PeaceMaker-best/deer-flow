"""Text-block tool results participate in the same stagnation guard as strings."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deerflow.agents.middlewares.tool_progress_middleware import ToolProgressMiddleware
from deerflow.agents.middlewares.tool_result_meta import TOOL_META_KEY, ToolResultMeta

TEXT = "alpha beta gamma delta epsilon zeta deployment memory preference"
META = asdict(ToolResultMeta(status="success", error_type=None, recoverable_by_model=True, recommended_next_action="continue", source="tool_return"))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("content", [[{"type": "text", "text": TEXT}], [TEXT], [{"type": "image", "base64": "opaque"}, {"type": "text", "text": TEXT}]])
async def test_repeated_text_blocks_trigger_warning_without_mutating_results(content, asynchronous):
    middleware = ToolProgressMiddleware(stagnation_threshold=2, min_words=3)
    runtime = SimpleNamespace(context={"thread_id": "thread", "run_id": "run"})
    middleware.before_agent({}, runtime)
    message = ToolMessage(content=content, tool_call_id="call", additional_kwargs={TOOL_META_KEY: META})
    request = SimpleNamespace(tool_call={"name": "lookup", "id": "call"}, runtime=runtime)

    async def handler(_request):
        return message

    for _ in range(3):
        returned = await middleware.awrap_tool_call(request, handler) if asynchronous else middleware.wrap_tool_call(request, lambda _: message)
        assert returned is message
    assert middleware._drain_pending(runtime)
    assert message.content == content


def test_media_only_results_do_not_invent_textual_similarity():
    middleware = ToolProgressMiddleware(stagnation_threshold=1, min_words=1)
    runtime = SimpleNamespace(context={"thread_id": "thread", "run_id": "run"})
    request = SimpleNamespace(tool_call={"name": "lookup", "id": "call"}, runtime=runtime)
    message = ToolMessage(content=[{"type": "image", "base64": TEXT}], tool_call_id="call", additional_kwargs={TOOL_META_KEY: META})
    for _ in range(4):
        middleware.wrap_tool_call(request, lambda _: message)
    assert middleware._drain_pending(runtime) == []
