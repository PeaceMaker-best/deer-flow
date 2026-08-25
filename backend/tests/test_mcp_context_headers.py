"""Tests for request-scoped MCP headers sourced from run context secrets."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import ToolException
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from pydantic import ValidationError

from deerflow.config.extensions_config import ExtensionsConfig, McpServerConfig
from deerflow.mcp.context_headers import build_context_headers_interceptor
from deerflow.mcp.interceptors import build_mcp_tool_interceptors


def _config(*, headers_from_context: dict[str, str], transport: str = "http", enabled: bool = True) -> ExtensionsConfig:
    return ExtensionsConfig(
        mcp_servers={
            "tenant-api": McpServerConfig(
                enabled=enabled,
                type=transport,
                url="https://mcp.example.com/mcp" if transport != "stdio" else None,
                command="mcp-server" if transport == "stdio" else None,
                headers_from_context=headers_from_context,
            ),
            "other": McpServerConfig(type="http", url="https://other.example.com/mcp"),
        },
        skills={},
    )


def _request(
    *,
    server_name: str = "tenant-api",
    headers: dict[str, str] | None = None,
    secrets: dict[str, object] | None = None,
) -> MCPToolCallRequest:
    runtime = None if secrets is None else SimpleNamespace(context={"secrets": secrets})
    return MCPToolCallRequest(
        name="search",
        args={},
        server_name=server_name,
        headers=headers,
        runtime=runtime,
    )


async def _echo_handler(request: MCPToolCallRequest) -> MCPToolCallRequest:
    return request


def test_no_context_header_mappings_returns_none():
    assert build_context_headers_interceptor(_config(headers_from_context={})) is None


def test_each_call_uses_its_own_runtime_secrets_and_preserves_other_headers():
    interceptor = build_context_headers_interceptor(
        _config(
            headers_from_context={
                "Authorization": "MCP_AUTHORIZATION",
                "X-Organization": "MCP_ORGANIZATION",
            }
        )
    )

    first = asyncio.run(
        interceptor(
            _request(
                headers={"Accept": "application/json", "authorization": "Bearer discovery"},
                secrets={"MCP_AUTHORIZATION": "Bearer tenant-a", "MCP_ORGANIZATION": "org-a"},
            ),
            _echo_handler,
        )
    )
    second = asyncio.run(
        interceptor(
            _request(
                secrets={"MCP_AUTHORIZATION": "Bearer tenant-b", "MCP_ORGANIZATION": "org-b"},
            ),
            _echo_handler,
        )
    )

    assert first.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer tenant-a",
        "X-Organization": "org-a",
    }
    assert second.headers == {
        "Authorization": "Bearer tenant-b",
        "X-Organization": "org-b",
    }


def test_missing_or_non_string_secret_fails_closed_without_calling_handler():
    interceptor = build_context_headers_interceptor(_config(headers_from_context={"Authorization": "TOKEN", "X-Organization": "ORG"}))
    handler = AsyncMock()

    with pytest.raises(ToolException, match="ORG") as exc_info:
        asyncio.run(interceptor(_request(secrets={"TOKEN": "Bearer private", "ORG": 42}), handler))

    handler.assert_not_awaited()
    assert "Bearer private" not in str(exc_info.value)


def test_ambient_run_context_is_used_when_adapter_runtime_is_unavailable():
    interceptor = build_context_headers_interceptor(_config(headers_from_context={"X-Tenant": "TENANT"}))
    with patch(
        "deerflow.mcp.context_headers._current_run_context",
        return_value={"secrets": {"TENANT": "tenant-from-config"}},
    ):
        result = asyncio.run(interceptor(_request(), _echo_handler))
    assert result.headers == {"X-Tenant": "tenant-from-config"}


def test_other_server_passes_through_untouched():
    interceptor = build_context_headers_interceptor(_config(headers_from_context={"X-Tenant": "TENANT"}))
    request = _request(server_name="other", headers={"X-Static": "value"}, secrets={"TENANT": "tenant-a"})
    assert asyncio.run(interceptor(request, _echo_handler)) is request


def test_disabled_server_is_ignored():
    assert build_context_headers_interceptor(_config(headers_from_context={"X-Tenant": "TENANT"}, enabled=False)) is None


def test_stdio_mapping_is_ignored_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="deerflow.mcp.context_headers"):
        interceptor = build_context_headers_interceptor(_config(headers_from_context={"X-Tenant": "TENANT"}, transport="stdio"))
    assert interceptor is None
    assert any("headers_from_context" in record.message and "stdio" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"Bad Header": "TOKEN"}, "header name"),
        ({"X-Tenant": "   "}, "secret key"),
        ({"Authorization": "TOKEN", "authorization": "OTHER"}, "case-insensitive"),
    ],
)
def test_invalid_context_header_mapping_is_rejected(mapping, message):
    with pytest.raises(ValidationError, match=message):
        McpServerConfig(type="http", url="https://mcp.example.com/mcp", headers_from_context=mapping)


def test_context_headers_cannot_be_combined_with_durable_task_toolsets():
    with pytest.raises(ValidationError, match="request-scoped.*durable"):
        McpServerConfig(
            type="http",
            url="https://mcp.example.com/mcp",
            headers_from_context={"Authorization": "TOKEN"},
            task_toolsets=[
                {
                    "name": "render",
                    "submit_tool": "submit",
                    "status_tool": "status",
                    "cancel_tool": "cancel",
                }
            ],
        )


def test_shared_assembly_registers_context_headers_after_builtin_auth():
    config = _config(headers_from_context={"Authorization": "TOKEN"})

    async def oauth(request, handler):
        return await handler(request)

    async def user_auth(request, handler):
        return await handler(request)

    async def context_headers(request, handler):
        return await handler(request)

    interceptors = build_mcp_tool_interceptors(
        config,
        oauth_builder=lambda _cfg: oauth,
        user_auth_builder=lambda _cfg: user_auth,
        context_headers_builder=lambda _cfg: context_headers,
    )
    assert interceptors == [oauth, user_auth, context_headers]


def test_gateway_schema_round_trips_context_header_mapping():
    from app.gateway.routers.mcp import McpServerConfigResponse

    server = McpServerConfigResponse(
        type="http",
        url="https://mcp.example.com/mcp",
        headers_from_context={"Authorization": "TOKEN"},
    )
    assert server.headers_from_context == {"Authorization": "TOKEN"}
    assert McpServerConfigResponse().headers_from_context == {}


def test_gateway_schema_rejects_context_headers_for_durable_tasks():
    from app.gateway.routers.mcp import McpServerConfigResponse

    with pytest.raises(ValidationError, match="request-scoped.*durable"):
        McpServerConfigResponse(
            type="http",
            url="https://mcp.example.com/mcp",
            headers_from_context={"Authorization": "TOKEN"},
            task_toolsets=[
                {
                    "name": "render",
                    "submit_tool": "submit",
                    "status_tool": "status",
                    "cancel_tool": "cancel",
                }
            ],
        )
