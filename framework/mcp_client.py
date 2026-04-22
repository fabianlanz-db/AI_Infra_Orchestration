"""MCP client for AI Gateway-governed Databricks MCP servers.

Uses the official ``mcp`` Python SDK with the Streamable HTTP transport
per the MCP specification. Databricks Unity AI Gateway governs MCP
invocations via Unity Catalog permissions, UC connections for external
credentials, and centralized audit — the invocation itself is a plain
Bearer-authenticated HTTPS call against the server's own URL.

Auth strategies are defined as a small Protocol; concrete U2M / M2M /
PAT implementations live in ``framework.mcp_auth``. Callers who need
non-standard auth (e.g. a pre-authorized ``httpx.Auth`` or an existing
OAuth session) can pass a ``session_factory`` instead — the framework
then delegates session construction entirely (BYO mode).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from framework.mcp_catalog_utils import MCPToolDefinition, MCPToolResult
from framework.mlflow_tracing_utils import set_mcp_tags


@runtime_checkable
class MCPAuth(Protocol):
    """Supplies Bearer tokens for MCP calls."""

    def bearer_token(self) -> str:
        """Return a valid Bearer token; implementations handle refresh."""

    def mode(self) -> str:
        """Short label for ``mcp.auth_mode`` tag: ``u2m`` | ``m2m`` | ``pat``."""


@dataclass
class MCPInvocation:
    """Everything needed to invoke one tool on one server."""

    server: str          # logical name from the catalog
    server_type: str     # "managed" | "external" | "custom"
    server_url: str      # HTTP URL of the MCP server
    tool: str
    arguments: dict[str, Any] | None = None


# Callers who BYO auth supply a factory that, given a URL, yields a
# ready-to-use (initialized) ClientSession.
SessionFactory = Callable[[str], AsyncIterator[ClientSession]]


class DatabricksMCPClient:
    """Invokes MCP tools via the official ``mcp`` SDK's Streamable HTTP transport.

    Either ``auth`` or ``session_factory`` must be supplied.

    Sync methods (``invoke_tool`` / ``list_tools``) are convenience wrappers
    around the async variants (``ainvoke_tool`` / ``alist_tools``). The sync
    wrappers raise ``RuntimeError`` if called from inside an active event
    loop — use the async variants in that case.
    """

    def __init__(
        self,
        auth: MCPAuth | None = None,
        session_factory: SessionFactory | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if auth is None and session_factory is None:
            raise ValueError(
                "DatabricksMCPClient requires either auth= (bearer-token strategy) "
                "or session_factory= (BYO session). Pass one."
            )
        self._auth = auth
        self._session_factory = session_factory
        self._timeout = timeout_seconds

    @property
    def auth_mode(self) -> str:
        """Label for ``mcp.auth_mode`` tagging and compat checks.

        Returns ``"byo"`` when a ``session_factory`` is supplied instead of
        ``auth``; otherwise delegates to the ``MCPAuth`` implementation.
        """
        return self._auth.mode() if self._auth else "byo"

    # ---- Sync entry points -------------------------------------------------

    def invoke_tool(self, invocation: MCPInvocation) -> MCPToolResult:
        """Invoke a tool. Blocks until the tool returns."""
        return _run_sync(self.ainvoke_tool(invocation))

    def list_tools(self, server_url: str) -> list[MCPToolDefinition]:
        """Discover tools exposed by a server. Blocks until the RPC returns."""
        return _run_sync(self.alist_tools(server_url))

    # ---- Async entry points ------------------------------------------------

    async def ainvoke_tool(self, invocation: MCPInvocation) -> MCPToolResult:
        start = time.perf_counter()
        auth_mode = self.auth_mode
        async with self._session(invocation.server_url) as session:
            response = await session.call_tool(
                invocation.tool,
                arguments=invocation.arguments,
            )
        latency_ms = int((time.perf_counter() - start) * 1000)
        set_mcp_tags(
            server=invocation.server,
            server_type=invocation.server_type,
            server_url=invocation.server_url,
            tool=invocation.tool,
            latency_ms=latency_ms,
            auth_mode=auth_mode,
        )
        # SDK returns a CallToolResult with .content / .structuredContent /
        # .isError — we pass it through so callers can branch on isError.
        return MCPToolResult(
            output=response,
            latency_ms=latency_ms,
            server_name=invocation.server,
            tool_name=invocation.tool,
        )

    async def alist_tools(self, server_url: str) -> list[MCPToolDefinition]:
        async with self._session(server_url) as session:
            response = await session.list_tools()
        # server_name is left blank here; the catalog stamps it on registration
        # since one client can list tools from many servers.
        return [
            MCPToolDefinition(
                name=t.name,
                description=t.description or "",
                server_name="",
                input_schema=t.inputSchema or {},
            )
            for t in response.tools
        ]

    # ---- Session management ------------------------------------------------

    @asynccontextmanager
    async def _session(self, server_url: str) -> AsyncIterator[ClientSession]:
        if self._session_factory is not None:
            async with self._session_factory(server_url) as session:  # type: ignore[union-attr]
                yield session
            return
        assert self._auth is not None  # guarded in __init__
        headers = {"Authorization": f"Bearer {self._auth.bearer_token()}"}
        async with streamablehttp_client(
            server_url, headers=headers, timeout=self._timeout,
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from sync code via ``asyncio.run``.

    Raises ``RuntimeError`` if an event loop is already running; callers
    inside an async context must use the async variants instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside an event loop — close the coroutine so Python doesn't warn
    # about "coroutine was never awaited".
    coro.close()
    raise RuntimeError(
        "DatabricksMCPClient sync methods cannot run inside an active event "
        "loop — call the async variants (ainvoke_tool, alist_tools) instead."
    )
