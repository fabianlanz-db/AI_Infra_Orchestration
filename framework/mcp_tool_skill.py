"""Adapter that exposes an MCP tool through the ``SkillClient`` protocol.

Extracted from ``mcp_catalog_utils`` so the catalog module stays under the
300-line gate. ``_MCPToolSkill`` is intentionally private — consumers get
these indirectly via ``MCPCatalogClient.sync_to_skill_registry``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from framework.skill_registry import SkillDefinition, SkillInput, SkillResult

if TYPE_CHECKING:
    from framework.mcp_catalog_utils import MCPToolDefinition, MCPToolResult


# Skill-side invoker: given a SkillInput, returns the MCPToolResult from
# the server. The catalog closes over its client + server config to build
# these, keeping the skill itself decoupled from the transport.
_ToolInvoker = Callable[[SkillInput], "MCPToolResult"]


class _MCPToolSkill:
    """Internal adapter: wraps an ``MCPToolDefinition`` as a ``SkillClient``.

    If constructed with ``invoke_fn``, ``execute()`` dispatches to it (the
    catalog builds these once a ``DatabricksMCPClient`` is attached). Without
    an invoker, ``execute()`` raises — the catalog is pure metadata.
    """

    def __init__(
        self,
        tool: MCPToolDefinition,
        invoke_fn: _ToolInvoker | None = None,
    ) -> None:
        self._tool = tool
        self._invoke = invoke_fn

    @property
    def name(self) -> str:
        return f"mcp:{self._tool.server_name}:{self._tool.name}"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description=self._tool.description or f"MCP tool {self._tool.name}",
            tags=["mcp", self._tool.server_name],
            input_schema=self._tool.input_schema,
            source="mcp",
            mcp_server=self._tool.server_name,
        )

    def execute(self, input: SkillInput) -> SkillResult:
        if self._invoke is None:
            raise NotImplementedError(
                f"MCP tool '{self._tool.name}' on server '{self._tool.server_name}' "
                f"has no invoker. Attach a DatabricksMCPClient via "
                f"MCPCatalogClient.set_client() before sync_to_skill_registry()."
            )
        tool_result = self._invoke(input)
        return SkillResult(
            output=tool_result.output,
            latency_ms=tool_result.latency_ms,
            skill_name=self.name,
            metadata={
                "mcp_server": self._tool.server_name,
                "mcp_tool": self._tool.name,
            },
        )

    def health(self) -> tuple[bool, str]:
        if self._invoke is None:
            return True, f"MCP tool {self._tool.name} registered (metadata only)"
        return True, f"MCP tool {self._tool.name} registered and executable"
