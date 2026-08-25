"""Request-scoped HTTP headers for multi-tenant MCP tool calls.

An HTTP/SSE server can map header names to keys under
``config.context.secrets`` with ``headers_from_context``. Values are resolved
for every tool call and applied through the adapter's per-request header
override, so concurrent tenants never share a cached credential or routing
header. Missing values fail closed and secret values are never logged.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import ToolException

from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.runtime.secret_context import extract_request_secrets

logger = logging.getLogger(__name__)


def _current_run_context() -> Any | None:
    """Best-effort access to LangGraph config when no runtime is attached."""
    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return None
    if isinstance(config, dict):
        return config.get("context")
    return None


def build_context_headers_interceptor(extensions_config: ExtensionsConfig) -> Any | None:
    """Build a per-call context-header interceptor, or ``None`` if unused."""
    mappings_by_server: dict[str, dict[str, str]] = {}
    for server_name, server_config in extensions_config.get_enabled_mcp_servers().items():
        if not server_config.headers_from_context:
            continue
        if server_config.type not in ("sse", "http"):
            logger.warning(
                "MCP server '%s' declares headers_from_context but uses the '%s' transport; request-scoped headers only apply to 'sse'/'http' servers — ignoring headers_from_context for this server",
                server_name,
                server_config.type,
            )
            continue
        mappings_by_server[server_name] = server_config.headers_from_context

    if not mappings_by_server:
        return None

    async def context_headers_interceptor(request: Any, handler: Any) -> Any:
        mapping = mappings_by_server.get(request.server_name)
        if mapping is None:
            return await handler(request)

        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None) if runtime is not None else None
        if not isinstance(context, dict):
            context = _current_run_context()
        secrets = extract_request_secrets(context)
        missing_keys = sorted({secret_key for secret_key in mapping.values() if not secrets.get(secret_key)})
        if missing_keys:
            logger.warning(
                "Denied MCP tool call to server '%s': missing request-scoped secret keys %s",
                request.server_name,
                missing_keys,
            )
            missing = ", ".join(missing_keys)
            raise ToolException(f"Missing request-scoped secret(s) {missing} required by headers_from_context for MCP server '{request.server_name}'. Pass them via config.context.secrets.")

        mapped_header_names = {header_name.casefold() for header_name in mapping}
        updated_headers = {header_name: value for header_name, value in (request.headers or {}).items() if header_name.casefold() not in mapped_header_names}
        for header_name, secret_key in mapping.items():
            updated_headers[header_name] = secrets[secret_key]
        return await handler(request.override(headers=updated_headers))

    return context_headers_interceptor
