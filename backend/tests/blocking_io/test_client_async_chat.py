"""Shared embedded turn preparation must not perform IO on the async loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from deerflow.client import DeerFlowClient
from deerflow.config.app_config import AppConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig


@pytest.mark.asyncio
async def test_achat_offloads_real_graph_setup_io(tmp_path):
    file = tmp_path / "agent-config.txt"
    await asyncio.to_thread(file.write_text, "graph setup", encoding="utf-8")
    config = AppConfig(models=[ModelConfig(name="test", model="test", use="langchain_openai:ChatOpenAI")], sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"))
    with patch("deerflow.client.get_app_config", return_value=config):
        client = await asyncio.to_thread(DeerFlowClient)
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="answer", id="current")]})

    def ensure_agent(config, *, context):
        assert file.read_text(encoding="utf-8") == "graph setup"
        client._agent = graph

    with (
        patch.object(client, "_ensure_agent", side_effect=ensure_agent),
        patch("deerflow.runtime.checkpointer.get_checkpointer", return_value=None),
    ):
        assert await client.achat("question") == "answer"
    graph.ainvoke.assert_awaited_once()
