"""Unit tests for framework.unified_catalog."""
from __future__ import annotations

import json

from framework.mcp_catalog_utils import (
    MCPCatalogClient,
    MCPServerConfig,
    MCPToolDefinition,
)
from framework.reference_uc_bindings import vector_search_binding
from framework.skill_registry import (
    SkillClient,
    SkillDefinition,
    SkillInput,
    SkillRegistry,
    SkillResult,
)
from framework.unified_catalog import build_unified_tool_catalog


class _LocalSkill:
    @property
    def name(self) -> str:
        return "local-skill"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name="local-skill",
            description="Some in-process skill.",
            version="1.0.0",
            tags=["x"],
            input_schema={"query": "str"},
            output_schema={"text": "str"},
            source="local",
        )

    def execute(self, input: SkillInput) -> SkillResult:
        return SkillResult(output={}, latency_ms=0, skill_name=self.name)

    def health(self) -> tuple[bool, str]:
        return True, "ok"


def test_empty_inputs_return_empty_sections():
    payload = build_unified_tool_catalog()
    assert payload["native_skills"] == []
    assert payload["uc_functions"] == []
    assert payload["mcp_tools"] == []
    assert "generated_at" in payload


def test_native_skills_included_and_mcp_skills_filtered_out():
    """MCP-sourced skills injected into a registry should NOT appear in
    native_skills — they show up under mcp_tools instead, so no double-count.
    """
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="srv", server_type="custom"))
    catalog.register_tool(MCPToolDefinition(name="t", description="d", server_name="srv"))
    registry = SkillRegistry()
    registry.register(_LocalSkill())
    catalog.sync_to_skill_registry(registry)  # injects an mcp:srv:t skill

    payload = build_unified_tool_catalog(skill_registry=registry, mcp_catalog=catalog)
    native_names = [s["name"] for s in payload["native_skills"]]
    assert native_names == ["local-skill"]


def test_uc_functions_reported_with_fq_name_and_columns():
    binding = vector_search_binding(
        catalog="main", schema="agents",
        serving_endpoint="https://host/serving-endpoints/vs/invocations",
    )
    payload = build_unified_tool_catalog(uc_bindings=[binding])
    funcs = payload["uc_functions"]
    assert len(funcs) == 1
    f = funcs[0]
    assert f["fq_name"] == "main.agents.vector_search"
    assert {c["name"] for c in f["input_columns"]} == {"query", "top_k"}


def test_mcp_tools_stamped_with_server_metadata():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(
        name="uc-funcs", server_type="managed",
        url="https://host/api/2.0/mcp/functions/cat/sch",
    ))
    catalog.register_tool(MCPToolDefinition(
        name="search", description="Search UC", server_name="uc-funcs",
        input_schema={"type": "object"},
    ))
    payload = build_unified_tool_catalog(mcp_catalog=catalog)
    tools = payload["mcp_tools"]
    assert len(tools) == 1
    t = tools[0]
    assert t["name"] == "search"
    assert t["server_type"] == "managed"
    assert t["server_url"] == "https://host/api/2.0/mcp/functions/cat/sch"


def test_full_payload_is_json_serialisable():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="s", server_type="managed", url="https://h/mcp"))
    catalog.register_tool(MCPToolDefinition(name="t", description="d", server_name="s"))
    registry = SkillRegistry()
    registry.register(_LocalSkill())
    binding = vector_search_binding(
        catalog="main", schema="agents",
        serving_endpoint="https://h/serving-endpoints/vs/invocations",
    )
    payload = build_unified_tool_catalog(
        skill_registry=registry, mcp_catalog=catalog, uc_bindings=[binding],
    )
    # Should round-trip through JSON without losses.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert set(decoded.keys()) == {
        "native_skills", "uc_functions", "mcp_tools", "generated_at",
    }
