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


# --- discover_tools ----------------------------------------------------------

class _FakeListingClient:
    """Minimal DatabricksMCPClient stand-in for discover_tools tests.

    Returns a preset list of MCPToolDefinition per URL. Tracks which URLs
    were queried so tests can assert on dispatch.
    """

    def __init__(self, tools_by_url: dict[str, list[MCPToolDefinition]]) -> None:
        self._tools = tools_by_url
        self.calls: list[str] = []
        # discover_tools checks client.auth_mode if it calls _make_invoker,
        # but discover_tools itself doesn't. Still, good to provide.
        self.auth_mode = "byo"

    def list_tools(self, server_url: str) -> list[MCPToolDefinition]:
        self.calls.append(server_url)
        return list(self._tools.get(server_url, []))


def test_discover_tools_raises_without_client():
    import pytest
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="s", server_type="managed", url="https://h/mcp"))
    with pytest.raises(RuntimeError, match="DatabricksMCPClient"):
        catalog.discover_tools()


def test_discover_tools_populates_catalog_and_stamps_server_name():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(
        name="uc-funcs", server_type="managed",
        url="https://h/api/2.0/mcp/functions/cat/sch",
    ))
    # list_tools returns MCPToolDefinition with server_name="" (matches the
    # contract in framework.mcp_client.alist_tools — catalog stamps on add).
    client = _FakeListingClient({
        "https://h/api/2.0/mcp/functions/cat/sch": [
            MCPToolDefinition(name="search", description="Search docs", server_name=""),
            MCPToolDefinition(name="retrieve", description="Get a doc", server_name=""),
        ],
    })
    added = catalog.discover_tools(client=client)
    assert added == 2
    tools = catalog.list_tools("uc-funcs")
    assert {t.name for t in tools} == {"search", "retrieve"}
    # Catalog stamped the server name
    assert all(t.server_name == "uc-funcs" for t in tools)


def test_discover_tools_dedupes_on_second_call():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="s", server_type="managed", url="https://h/mcp"))
    client = _FakeListingClient({
        "https://h/mcp": [MCPToolDefinition(name="t", description="d", server_name="")],
    })
    assert catalog.discover_tools(client=client) == 1
    # Second call: tool already known, so added=0.
    assert catalog.discover_tools(client=client) == 0


def test_discover_tools_restricts_to_named_server():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="a", server_type="managed", url="https://h/a"))
    catalog.register_server(MCPServerConfig(name="b", server_type="managed", url="https://h/b"))
    client = _FakeListingClient({
        "https://h/a": [MCPToolDefinition(name="ta", description="d", server_name="")],
        "https://h/b": [MCPToolDefinition(name="tb", description="d", server_name="")],
    })
    catalog.discover_tools(server_name="a", client=client)
    assert client.calls == ["https://h/a"]
    assert catalog.list_tools("a") and not catalog.list_tools("b")


def test_discover_tools_skips_disabled_servers():
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(
        name="on", server_type="managed", url="https://h/on", enabled=True,
    ))
    catalog.register_server(MCPServerConfig(
        name="off", server_type="managed", url="https://h/off", enabled=False,
    ))
    client = _FakeListingClient({
        "https://h/on": [MCPToolDefinition(name="t", description="d", server_name="")],
        "https://h/off": [MCPToolDefinition(name="t", description="d", server_name="")],
    })
    catalog.discover_tools(client=client)
    assert client.calls == ["https://h/on"]


def test_discover_tools_warns_when_server_has_no_url(caplog):
    import logging
    catalog = MCPCatalogClient()
    catalog.register_server(MCPServerConfig(name="s", server_type="custom"))  # no url
    client = _FakeListingClient({})
    with caplog.at_level(logging.WARNING, logger="framework.mcp_catalog_utils"):
        added = catalog.discover_tools(client=client)
    assert added == 0
    assert any("no URL" in r.message for r in caplog.records)


def test_discover_tools_unknown_server_name_raises():
    import pytest
    catalog = MCPCatalogClient()
    client = _FakeListingClient({})
    with pytest.raises(KeyError, match="nope"):
        catalog.discover_tools(server_name="nope", client=client)
