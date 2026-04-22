"""
MCP catalog utilities for managing managed, custom, and external MCP servers.

Provides MCPCatalogClient for discovering, registering, and querying MCP
servers and their tools. Bridges MCP tools into the SkillRegistry so any
agent runtime can discover them alongside native skills.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from framework.skill_registry import SkillRegistry

if TYPE_CHECKING:
    from framework.mcp_client import DatabricksMCPClient


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server (managed, custom, or external).

    ``url`` is the HTTP endpoint used by ``DatabricksMCPClient`` for Streamable
    HTTP invocation against AI Gateway-governed servers. ``command``/``args``
    remain for legacy stdio-style configs imported from ``.cursor/mcp.json``.
    """

    name: str
    server_type: str  # "managed" | "custom" | "external"
    url: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolDefinition:
    """Metadata for a single tool exposed by an MCP server."""

    name: str
    description: str
    server_name: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """Result from invoking an MCP tool."""

    output: Any
    latency_ms: int
    server_name: str
    tool_name: str


class MCPCatalogClient:
    """Registry for MCP servers and their tools.

    Supports three server types:
    - **managed**: Databricks-provided MCP servers.
    - **custom**: User-deployed MCP servers on Databricks infrastructure.
    - **external**: Third-party MCP servers running outside Databricks.

    Server configs can be loaded from ``.cursor/mcp.json`` files or
    registered programmatically. Tools can be bridged into a
    SkillRegistry for unified agent discovery.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, list[MCPToolDefinition]] = {}
        self._client: DatabricksMCPClient | None = None

    def set_client(self, client: DatabricksMCPClient) -> None:
        """Attach an MCP client for tool invocation.

        After calling this, ``sync_to_skill_registry`` will produce skills
        whose ``execute()`` method dispatches to the client. Without a client
        attached, skills raise ``NotImplementedError`` on execute — the catalog
        remains pure metadata.
        """
        self._client = client

    def register_server(self, config: MCPServerConfig) -> None:
        """Add or update an MCP server in the catalog."""
        self._servers[config.name] = config
        if config.name not in self._tools:
            self._tools[config.name] = []

    def unregister_server(self, name: str) -> None:
        """Remove a server and its tools from the catalog."""
        self._servers.pop(name, None)
        self._tools.pop(name, None)

    def register_tool(self, tool: MCPToolDefinition) -> None:
        """Register a tool under its server."""
        if tool.server_name not in self._tools:
            self._tools[tool.server_name] = []
        existing_names = {t.name for t in self._tools[tool.server_name]}
        if tool.name not in existing_names:
            self._tools[tool.server_name].append(tool)

    def list_servers(self, server_type: str | None = None) -> list[MCPServerConfig]:
        """Return all registered servers, optionally filtered by type."""
        servers = list(self._servers.values())
        if server_type:
            servers = [s for s in servers if s.server_type == server_type]
        return servers

    def list_tools(self, server_name: str | None = None) -> list[MCPToolDefinition]:
        """Return tools across all servers or for a specific server."""
        if server_name:
            return list(self._tools.get(server_name, []))
        all_tools: list[MCPToolDefinition] = []
        for tools in self._tools.values():
            all_tools.extend(tools)
        return all_tools

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Lookup a server by name."""
        return self._servers.get(name)

    def sync_from_mcp_json(self, path: str | Path) -> int:
        """Import server configs from a ``.cursor/mcp.json`` file.

        Returns the number of servers imported.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("MCP config file not found at %s; no servers imported", path)
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        count = 0
        for name, cfg in servers.items():
            server_type = cfg.get("server_type", "custom")
            config = MCPServerConfig(
                name=name,
                server_type=server_type,
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                enabled=not cfg.get("disabled", False),
                metadata={
                    k: v for k, v in cfg.items()
                    if k not in {"command", "args", "env", "disabled", "server_type"}
                },
            )
            self.register_server(config)
            count += 1
        return count

    def discover_tools(
        self,
        server_name: str | None = None,
        client: DatabricksMCPClient | None = None,
    ) -> int:
        """Query registered (enabled) servers for their tool lists.

        Calls the standard MCP ``list_tools`` RPC against each server URL
        via the supplied or attached :class:`DatabricksMCPClient`, and
        populates the catalog via :meth:`register_tool` (deduped by name).

        ``server_name`` restricts discovery to one server. ``client``
        overrides the catalog's attached client for this call.

        Returns the number of tools newly added (existing tools are not
        counted again).
        """
        invoker = client or self._client
        if invoker is None:
            raise RuntimeError(
                "discover_tools requires a DatabricksMCPClient — attach one "
                "via set_client() or pass client=...",
            )
        if server_name is not None:
            server = self._servers.get(server_name)
            if server is None:
                raise KeyError(f"Server {server_name!r} not registered")
            servers = [server] if server.enabled else []
        else:
            servers = [s for s in self._servers.values() if s.enabled]

        added = 0
        for server in servers:
            if not server.url:
                logger.warning(
                    "Server %s has no URL; skipping tool discovery", server.name,
                )
                continue
            existing = {t.name for t in self._tools.get(server.name, [])}
            for tool in invoker.list_tools(server.url):
                tool.server_name = server.name
                self.register_tool(tool)
                if tool.name not in existing:
                    added += 1
        return added

    def sync_to_skill_registry(self, registry: SkillRegistry) -> int:
        """Bridge all MCP tools into a SkillRegistry as ``source='mcp'`` skills.

        If a client is attached via :meth:`set_client`, registered skills are
        executable. Otherwise they raise ``NotImplementedError`` on execute.

        Returns the number of skills registered.
        """
        from framework.mcp_tool_skill import _MCPToolSkill  # local to avoid cycle

        count = 0
        for tool in self.list_tools():
            server = self._servers.get(tool.server_name)
            if server and not server.enabled:
                continue
            invoker = self._make_invoker(tool, server) if (server and self._client) else None
            registry.register(_MCPToolSkill(tool=tool, invoke_fn=invoker))
            count += 1
        return count

    def _make_invoker(
        self, tool: MCPToolDefinition, server: MCPServerConfig,
    ) -> "_ToolInvoker":
        """Close over (client, server, tool) to produce a per-skill invoker.

        Enforces auth/server-type compatibility at registration time
        (e.g. custom MCP rejects PAT per the Databricks docs) so misconfig
        fails loudly here rather than at the first tool call.
        """
        # Local imports avoid cycles between mcp_catalog_utils and mcp_client /
        # mcp_tool_skill.
        from framework.mcp_auth import ensure_auth_compatible_with_server
        from framework.mcp_client import MCPInvocation
        from framework.skill_registry import SkillInput

        client = self._client
        assert client is not None  # caller guards
        ensure_auth_compatible_with_server(client.auth_mode, server.server_type)

        def invoke(skill_input: SkillInput) -> MCPToolResult:
            return client.invoke_tool(
                MCPInvocation(
                    server=server.name,
                    server_type=server.server_type,
                    server_url=server.url,
                    tool=tool.name,
                    arguments=skill_input.parameters,
                ),
            )

        return invoke

    def health(self) -> tuple[bool, str]:
        server_count = len(self._servers)
        tool_count = sum(len(t) for t in self._tools.values())
        return True, f"MCP catalog: {server_count} server(s), {tool_count} tool(s)"

    def build_external_mcp_catalog_payload(self) -> dict[str, Any]:
        """Export the MCP catalog for external agents."""
        servers = []
        for s in self._servers.values():
            servers.append({
                "name": s.name, "server_type": s.server_type,
                "enabled": s.enabled, "tool_count": len(self._tools.get(s.name, [])),
            })
        tools = []
        for t in self.list_tools():
            tools.append({
                "name": t.name, "description": t.description,
                "server_name": t.server_name, "input_schema": t.input_schema,
            })
        return {"server_count": len(servers), "servers": servers, "tools": tools}


# _MCPToolSkill has moved to framework.mcp_tool_skill to keep this module
# under the 300-line gate. sync_to_skill_registry imports it locally.
