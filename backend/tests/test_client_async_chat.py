"""Native asynchronous embedded conversations, without a sync-chat adapter."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deerflow.client import DeerFlowClient
from deerflow.config.app_config import AppConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.trace_context import get_current_trace_id, request_trace_context


@pytest.fixture
def async_client():
    config = AppConfig(models=[ModelConfig(name="test", model="test", use="langchain_openai:ChatOpenAI")], sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"))
    with patch("deerflow.client.get_app_config", return_value=config):
        client = DeerFlowClient()
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={"messages": []}))
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="answer", id="current")]})
    client._agent = graph
    with (
        patch.object(client, "_ensure_agent", return_value=graph),
        patch.object(client, "chat", side_effect=AssertionError("Sync chat must not run")),
        patch.object(client, "stream", side_effect=AssertionError("Sync stream must not run")),
        patch("deerflow.runtime.checkpointer.get_checkpointer", return_value=None),
        patch("deerflow.sandbox.lease.release_sandbox_execution_lease_async", new_callable=AsyncMock) as cleanup,
    ):
        yield client, graph, cleanup


@pytest.mark.asyncio
async def test_achat_awaits_native_graph_and_preserves_run_context(async_client):
    client, graph, cleanup = async_client
    assert await client.achat("question", thread_id="async-thread", user_id="alice", recursion_limit=17) == "answer"
    graph.ainvoke.assert_awaited_once()
    call = graph.ainvoke.call_args
    assert isinstance(call.args[0]["messages"][0], HumanMessage)
    assert call.args[0]["messages"][0].content == "question"
    assert call.kwargs["config"]["recursion_limit"] == 17
    assert call.kwargs["context"]["user_id"] == "alice"
    assert call.kwargs["context"]["thread_id"] == "async-thread"
    assert call.kwargs["context"]["run_id"]
    cleanup.assert_awaited_once_with(call.kwargs["context"])


@pytest.mark.asyncio
async def test_achat_inherits_trace_and_restores_caller_context(async_client):
    client, graph, cleanup = async_client
    observed = []

    async def invoke(*args, **kwargs):
        observed.append(get_current_trace_id())
        return {"messages": [AIMessage(content="answer", id="current")]}

    graph.ainvoke.side_effect = invoke
    previous = get_current_trace_id()
    with request_trace_context("caller-trace"):
        assert await client.achat("question") == "answer"
        assert get_current_trace_id() == "caller-trace"
    assert observed == ["caller-trace"]
    assert get_current_trace_id() == previous


@pytest.mark.asyncio
async def test_achat_does_not_return_unchanged_checkpoint_history(async_client):
    client, graph, cleanup = async_client
    client._checkpointer = MagicMock(aget_tuple=AsyncMock(return_value=None))
    historical = AIMessage(content="old answer")
    graph.aget_state.return_value = SimpleNamespace(values={"messages": [historical]})

    async def invoke(state, **kwargs):
        return {"messages": [historical, *state["messages"], AIMessage(content="", id="current")]}

    graph.ainvoke.side_effect = invoke
    assert await client.achat("question") == ""


@pytest.mark.asyncio
async def test_achat_cancellation_releases_lease_and_restores_trace(async_client):
    client, graph, cleanup = async_client
    started = asyncio.Event()
    never_complete = asyncio.Event()

    async def invoke(*args, **kwargs):
        started.set()
        await never_complete.wait()

    graph.ainvoke.side_effect = invoke
    previous = get_current_trace_id()
    task = asyncio.create_task(client.achat("question"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cleanup.assert_awaited_once()
    assert get_current_trace_id() == previous


@pytest.mark.asyncio
async def test_achat_preserves_original_graph_error(async_client):
    client, graph, cleanup = async_client
    graph.ainvoke.side_effect = ValueError("model failed")
    cleanup.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError, match="model failed"):
        await client.achat("question")
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_achat_cancellation_drains_turn_construction(async_client):
    client, graph, cleanup = async_client
    started, release = threading.Event(), threading.Event()

    def setup(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)

    with patch.object(client, "_ensure_agent", side_effect=setup):
        task = asyncio.create_task(client.achat("question"))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_achat_concurrent_turns_capture_their_user_graph(async_client):
    client, unused_graph, cleanup = async_client
    graphs = {user: MagicMock(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content=user)]})) for user in ("alice", "bob")}

    def setup(config, *, context):
        client._agent = graphs[context["user_id"]]

    with patch.object(client, "_ensure_agent", side_effect=setup):
        assert await asyncio.gather(client.achat("one", user_id="alice"), client.achat("two", user_id="bob")) == ["alice", "bob"]
    for user, graph in graphs.items():
        assert graph.ainvoke.call_args.kwargs["context"]["user_id"] == user


class AsyncOnlyModel(BaseChatModel):
    """Offline model that fails if the client takes a synchronous graph path."""

    @property
    def _llm_type(self) -> str:
        return "async-only-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("Synchronous model execution is forbidden")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"async:{messages[-1].content}"))])


class AsyncToolModel(AsyncOnlyModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content=messages[-1].content)
        else:
            message = AIMessage(content="", tool_calls=[{"name": "async_echo", "args": {"value": "native tool"}, "id": "call-async", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.asyncio
async def test_achat_awaits_real_async_tool_in_native_agent_loop(async_client):
    client, unused_graph, cleanup = async_client
    observed = []

    @tool
    async def async_echo(value: str) -> str:
        """Return upper-case text through a native asynchronous tool call."""
        await asyncio.sleep(0)
        observed.append(value)
        return value.upper()

    client._agent = create_agent(model=AsyncToolModel(), tools=[async_echo])
    assert await client.achat("use the async tool") == "NATIVE TOOL"
    assert observed == ["native tool"]
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_achat_runs_real_graph_with_async_only_model_and_checkpoint(async_client):
    client, unused_graph, cleanup = async_client
    saver = InMemorySaver()
    client._checkpointer = saver
    client._agent = create_agent(model=AsyncOnlyModel(), checkpointer=saver)
    assert await client.achat("first", thread_id="native-async") == "async:first"
    assert await client.achat("second", thread_id="native-async") == "async:second"
    snapshot = await client._agent.aget_state(client._get_runnable_config("native-async"))
    assert [msg.content for msg in snapshot.values["messages"]] == ["first", "async:first", "second", "async:second"]
    assert cleanup.await_count == 2


@pytest.mark.asyncio
async def test_achat_rejects_sync_only_checkpoint_before_graph_execution(async_client):
    client, graph, cleanup = async_client
    client._checkpointer = MagicMock(aget_tuple=AsyncMock(side_effect=NotImplementedError))
    with pytest.raises(ValueError, match="async-capable checkpointer"):
        await client.achat("question")
    graph.ainvoke.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_achat_preserves_caller_owned_async_sqlite_saver(async_client):
    client, unused_graph, cleanup = async_client
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        client._checkpointer = saver
        client._agent = create_agent(model=AsyncOnlyModel(), checkpointer=saver)
        assert await client.achat("first", thread_id="sqlite-async") == "async:first"
        assert await client.achat("second", thread_id="sqlite-async") == "async:second"
        # The client must not close a caller-owned connection after a turn.
        checkpoint = await saver.aget_tuple(client._get_runnable_config("sqlite-async"))
        assert checkpoint is not None
        assert len(checkpoint.checkpoint["channel_values"]["messages"]) == 4
