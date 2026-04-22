"""Single-endpoint view of every tool an agent can call.

Stitches together three sources of tools:

* **native_skills** — in-process ``SkillClient`` implementations registered
  with a ``SkillRegistry``. Source = ``local``.
* **uc_functions** — skills published as Unity Catalog Functions via
  ``framework.uc_function_publisher``. Surfaced here from the declared
  ``SkillUCBinding`` list; the framework does not enumerate UC.
* **mcp_tools** — tools exposed by MCP servers in a ``MCPCatalogClient``.

Useful for external agents (Pattern A): fetch this payload once at session
start, hand the LLM a flat list, and let it route. The payload is JSON-
serialisable end-to-end.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from framework.mcp_catalog_utils import MCPCatalogClient
    from framework.skill_registry import SkillRegistry
    from framework.uc_function_publisher import SkillUCBinding


def build_unified_tool_catalog(
    skill_registry: SkillRegistry | None = None,
    mcp_catalog: MCPCatalogClient | None = None,
    uc_bindings: list[SkillUCBinding] | None = None,
) -> dict[str, Any]:
    """Produce one JSON-serialisable tool catalog spanning all three sources.

    Any of the three inputs may be ``None`` — the corresponding key is
    returned empty. Native skills are filtered to ``source="local"`` so MCP-
    sourced entries (injected into a registry by ``sync_to_skill_registry``)
    don't appear twice.
    """
    return {
        "native_skills": _native_skills(skill_registry),
        "uc_functions": _uc_functions(uc_bindings),
        "mcp_tools": _mcp_tools(mcp_catalog),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _native_skills(registry: SkillRegistry | None) -> list[dict[str, Any]]:
    if registry is None:
        return []
    return [
        {
            "name": d.name,
            "description": d.description,
            "version": d.version,
            "tags": list(d.tags),
            "input_schema": dict(d.input_schema),
            "output_schema": dict(d.output_schema),
            "source": d.source,
        }
        for d in registry.list_skills()
        if d.source == "local"
    ]


def _uc_functions(bindings: list[SkillUCBinding] | None) -> list[dict[str, Any]]:
    if not bindings:
        return []
    out = []
    for b in bindings:
        out.append({
            "skill_name": b.skill_name,
            "fq_name": f"{b.catalog}.{b.schema}.{b.function_name}",
            "catalog": b.catalog,
            "schema": b.schema,
            "function_name": b.function_name,
            "return_type": b.return_type,
            "serving_endpoint": b.serving_endpoint,
            "input_columns": [
                {
                    "name": c.name,
                    "uc_type": c.uc_type,
                    "comment": c.comment,
                    "default": c.default,
                }
                for c in b.input_columns
            ],
            "comment": b.comment,
        })
    return out


def _mcp_tools(catalog: MCPCatalogClient | None) -> list[dict[str, Any]]:
    if catalog is None:
        return []
    out = []
    # Group tools under their server for easy pivoting by server.
    for tool in catalog.list_tools():
        server = catalog.get_server(tool.server_name)
        out.append({
            "name": tool.name,
            "description": tool.description,
            "server_name": tool.server_name,
            "server_type": server.server_type if server else "unknown",
            "server_url": server.url if server else "",
            "input_schema": dict(tool.input_schema),
        })
    return out
