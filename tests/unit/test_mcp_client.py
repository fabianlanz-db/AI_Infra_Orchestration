"""Unit tests for framework.mcp_client.

Uses the BYO ``session_factory`` hook to inject a fake ClientSession, so
tests don't need to mock the official mcp SDK's transport internals.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from framework.mcp_client import (
    DatabricksMCPClient,
    MCPAuth,
    MCPInvocation,
)


class _FakeAuth:
    """MCPAuth stub returning a fixed token and mode."""

    def __init__(self, token: str = "test-token", mode: str = "m2m") -> None:
        self._token = token
        self._mode = mode

    def bearer_token(self) -> str:
        return self._token

    def mode(self) -> str:
        return self._mode


def _fake_tool_result(text: str = "ok") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=False,
    )


def _fake_list_result(*names: str) -> ListToolsResult:
    return ListToolsResult(
        tools=[Tool(name=n, description=f"desc of {n}", inputSchema={"type": "object"}) for n in names],
    )


def _session_factory_returning(
    call_tool_result: Any | None = None,
    list_tools_result: Any | None = None,
) -> tuple[Any, AsyncMock]:
    """Build a session_factory that yields a fake ClientSession.

    Returns (factory, session_mock) so tests can assert on session calls.
    """
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = call_tool_result or _fake_tool_result()
    session.list_tools.return_value = list_tools_result or _fake_list_result("t1")

    @asynccontextmanager
    async def factory(_url: str):
        yield session

    return factory, session


# --- Construction ------------------------------------------------------------

def test_client_requires_auth_or_session_factory():
    with pytest.raises(ValueError, match="auth= .* session_factory="):
        DatabricksMCPClient()


def test_client_accepts_auth():
    assert DatabricksMCPClient(auth=_FakeAuth()) is not None


def test_client_accepts_session_factory():
    factory, _ = _session_factory_returning()
    assert DatabricksMCPClient(session_factory=factory) is not None


# --- Async tool invocation ---------------------------------------------------

def test_ainvoke_tool_delegates_to_session():
    factory, session = _session_factory_returning()
    client = DatabricksMCPClient(session_factory=factory)
    inv = MCPInvocation(
        server="uc-funcs", server_type="managed",
        server_url="https://host/api/2.0/mcp/functions/cat/sch",
        tool="search", arguments={"q": "widgets"},
    )
    result = asyncio.run(client.ainvoke_tool(inv))
    session.call_tool.assert_awaited_once_with("search", arguments={"q": "widgets"})
    assert result.server_name == "uc-funcs"
    assert result.tool_name == "search"
    assert result.latency_ms >= 0


def test_alist_tools_converts_sdk_response():
    factory, session = _session_factory_returning(
        list_tools_result=_fake_list_result("search", "retrieve"),
    )
    client = DatabricksMCPClient(session_factory=factory)
    tools = asyncio.run(client.alist_tools("https://host/mcp"))
    session.list_tools.assert_awaited_once()
    assert [t.name for t in tools] == ["search", "retrieve"]
    assert all(t.input_schema == {"type": "object"} for t in tools)
    # server_name is left blank; catalog stamps on registration
    assert all(t.server_name == "" for t in tools)


# --- Tag emission ------------------------------------------------------------

def test_ainvoke_tool_emits_mcp_tags_with_auth_mode():
    factory, _ = _session_factory_returning()
    client = DatabricksMCPClient(session_factory=factory, auth=_FakeAuth(mode="m2m"))
    inv = MCPInvocation(
        server="uc-funcs", server_type="managed",
        server_url="https://host/mcp", tool="search",
    )
    with patch("framework.mcp_client.set_mcp_tags") as mock_tags:
        asyncio.run(client.ainvoke_tool(inv))
    assert mock_tags.call_count == 1
    _, kwargs = mock_tags.call_args
    assert kwargs["server"] == "uc-funcs"
    assert kwargs["server_type"] == "managed"
    assert kwargs["server_url"] == "https://host/mcp"
    assert kwargs["tool"] == "search"
    assert kwargs["auth_mode"] == "m2m"
    assert kwargs["latency_ms"] >= 0


def test_ainvoke_tool_byo_session_tags_auth_mode_as_byo():
    # When only a session_factory is supplied, auth_mode should be "byo".
    factory, _ = _session_factory_returning()
    client = DatabricksMCPClient(session_factory=factory)
    inv = MCPInvocation(
        server="x", server_type="custom", server_url="https://host/mcp", tool="t",
    )
    with patch("framework.mcp_client.set_mcp_tags") as mock_tags:
        asyncio.run(client.ainvoke_tool(inv))
    assert mock_tags.call_args.kwargs["auth_mode"] == "byo"


# --- Sync wrappers -----------------------------------------------------------

def test_invoke_tool_sync_works_from_sync_context():
    factory, session = _session_factory_returning()
    client = DatabricksMCPClient(session_factory=factory)
    result = client.invoke_tool(MCPInvocation(
        server="s", server_type="managed", server_url="https://host/mcp", tool="t",
    ))
    session.call_tool.assert_awaited_once()
    assert result.tool_name == "t"


def test_invoke_tool_sync_raises_from_async_context():
    factory, _ = _session_factory_returning()
    client = DatabricksMCPClient(session_factory=factory)

    async def _inner():
        with pytest.raises(RuntimeError, match="active event loop"):
            client.invoke_tool(MCPInvocation(
                server="s", server_type="managed",
                server_url="https://host/mcp", tool="t",
            ))

    asyncio.run(_inner())


# --- Catalog integration: _MCPToolSkill now executable when client attached --

def test_catalog_skill_raises_without_client_attached():
    from framework.mcp_catalog_utils import (
        MCPCatalogClient, MCPServerConfig, MCPToolDefinition,
    )
    from framework.skill_registry import SkillInput, SkillRegistry

    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="s", server_type="managed", url="https://h/mcp"))
    catalog.register_tool(MCPToolDefinition(name="t", description="d", server_name="s"))
    registry = SkillRegistry()
    catalog.sync_to_skill_registry(registry)
    skill = registry.get("mcp:s:t")
    assert skill is not None
    with pytest.raises(NotImplementedError, match="no invoker"):
        skill.execute(SkillInput(query="q"))


def test_catalog_skill_executes_when_client_attached():
    from framework.mcp_catalog_utils import (
        MCPCatalogClient, MCPServerConfig, MCPToolDefinition,
    )
    from framework.skill_registry import SkillInput, SkillRegistry

    factory, session = _session_factory_returning(
        call_tool_result=_fake_tool_result("pong"),
    )
    client = DatabricksMCPClient(session_factory=factory)

    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(
        name="uc", server_type="managed", url="https://host/api/2.0/mcp/functions/c/s",
    ))
    catalog.register_tool(MCPToolDefinition(name="ping", description="d", server_name="uc"))
    catalog.set_client(client)

    registry = SkillRegistry()
    catalog.sync_to_skill_registry(registry)
    skill = registry.get("mcp:uc:ping")
    assert skill is not None

    result = skill.execute(SkillInput(query="q", parameters={"x": 1}))
    session.call_tool.assert_awaited_once_with("ping", arguments={"x": 1})
    assert result.skill_name == "mcp:uc:ping"
    assert result.metadata["mcp_server"] == "uc"
    assert result.metadata["mcp_tool"] == "ping"
