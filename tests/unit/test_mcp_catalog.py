import json

from framework.mcp_catalog_utils import (
    MCPCatalogClient,
    MCPServerConfig,
    MCPToolDefinition,
)
from framework.skill_registry import SkillRegistry


def test_register_and_list_servers():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="srv1", server_type="custom"))
    catalog.register_server(MCPServerConfig(name="srv2", server_type="managed"))
    assert {s.name for s in catalog.list_servers()} == {"srv1", "srv2"}
    assert [s.name for s in catalog.list_servers(server_type="managed")] == ["srv2"]


def test_register_tool_deduplicates():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="srv", server_type="custom"))
    catalog.register_tool(MCPToolDefinition(name="t1", description="d", server_name="srv"))
    catalog.register_tool(MCPToolDefinition(name="t1", description="d", server_name="srv"))
    assert len(catalog.list_tools("srv")) == 1


def test_sync_from_mcp_json_missing_returns_zero(tmp_path, caplog):
    catalog = MCPCatalogClient()
    import logging
    with caplog.at_level(logging.WARNING, logger="framework.mcp_catalog_utils"):
        n = catalog.sync_from_mcp_json(tmp_path / "nope.json")
    assert n == 0
    assert any("MCP config file not found" in r.message for r in caplog.records)


def test_sync_from_mcp_json_imports_servers(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({
        "mcpServers": {
            "alpha": {"command": "node", "args": ["-e"], "server_type": "custom"},
            "beta": {"command": "python", "disabled": True},
        }
    }))
    catalog = MCPCatalogClient()
    n = catalog.sync_from_mcp_json(path)
    assert n == 2
    alpha = catalog.get_server("alpha")
    beta = catalog.get_server("beta")
    assert alpha.command == "node" and alpha.enabled is True
    assert beta.enabled is False


def test_sync_to_skill_registry_skips_disabled_servers():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="on", server_type="custom", enabled=True))
    catalog.register_server(MCPServerConfig(name="off", server_type="custom", enabled=False))
    catalog.register_tool(MCPToolDefinition(name="t1", description="d", server_name="on"))
    catalog.register_tool(MCPToolDefinition(name="t2", description="d", server_name="off"))
    reg = SkillRegistry()
    n = catalog.sync_to_skill_registry(reg)
    assert n == 1
    assert reg.get("mcp:on:t1") is not None
    assert reg.get("mcp:off:t2") is None
