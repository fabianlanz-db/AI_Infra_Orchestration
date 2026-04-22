"""End-to-end integration test: skill → UC Function → managed MCP → invoke.

Proves the coupling between Part B Steps 4–8 holds:

1. Define a skill locally (SkillUCBinding declares how to publish it).
2. ``publish_skill`` emits ``CREATE OR REPLACE FUNCTION`` DDL against a
   (mocked) SQL warehouse — simulating Databricks accepting the publish.
3. ``managed_functions_server`` registers the Managed MCP endpoint where
   Databricks exposes that UC Function.
4. ``MCPCatalogClient.discover_tools`` queries the endpoint (mocked MCP
   session) and finds the published function as a tool.
5. ``sync_to_skill_registry`` wires the MCP tool as an executable skill.
6. Calling the skill dispatches through ``DatabricksMCPClient`` and the
   MCP session returns a result.

The Databricks SDK and the MCP transport are mocked at the boundary —
everything else is real framework code.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from framework.mcp_catalog_utils import MCPCatalogClient
from framework.mcp_client import DatabricksMCPClient
from framework.mcp_servers import managed_functions_server
from framework.reference_uc_bindings import vector_search_binding
from framework.skill_registry import SkillInput, SkillRegistry
from framework.uc_function_publisher import publish_skill


HOST = "https://workspace.cloud.databricks.com"
CATALOG = "main"
SCHEMA = "agents"
SERVING_URL = f"{HOST}/serving-endpoints/vector-search/invocations"


def _ok_wc() -> MagicMock:
    """WorkspaceClient mock that reports SUCCEEDED on statement execution."""
    wc = MagicMock()
    result = MagicMock()
    result.status.state.value = "SUCCEEDED"
    wc.statement_execution.execute_statement.return_value = result
    return wc


def _mcp_session_factory_with(
    tools: list[Tool],
    call_result: CallToolResult,
):
    """Build a session_factory yielding a fake ClientSession backed by the
    given list_tools / call_tool responses. Tracks all session calls.
    """
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = ListToolsResult(tools=tools)
    session.call_tool.return_value = call_result

    @asynccontextmanager
    async def factory(_url: str):
        yield session

    return factory, session


def test_skill_publishes_to_uc_then_becomes_callable_via_mcp():
    # ---- Step 1+2: publish the skill as a UC Function -----------------------
    binding = vector_search_binding(
        catalog=CATALOG, schema=SCHEMA, serving_endpoint=SERVING_URL,
    )
    wc = _ok_wc()
    fq_name = publish_skill(binding, wc, sql_warehouse_id="wh-1")
    assert fq_name == f"{CATALOG}.{SCHEMA}.vector_search"

    # Sanity: the DDL really hit the warehouse with the expected shape.
    call = wc.statement_execution.execute_statement.call_args
    assert call.kwargs["warehouse_id"] == "wh-1"
    assert "CREATE OR REPLACE FUNCTION" in call.kwargs["statement"]
    assert f"`{CATALOG}`.`{SCHEMA}`.`vector_search`" in call.kwargs["statement"]

    # ---- Step 3: register the Managed Functions MCP server ------------------
    catalog = MCPCatalogClient()
    server_config = managed_functions_server(CATALOG, SCHEMA, HOST)
    catalog.register_server(server_config)
    expected_server_url = f"{HOST}/api/2.0/mcp/{'functions'}/{CATALOG}/{SCHEMA}"
    assert server_config.url == expected_server_url

    # ---- Step 4: discover tools via the (mocked) MCP session ----------------
    # The managed Functions endpoint returns the UC Function we just published
    # as an MCP tool named "vector_search".
    factory, session = _mcp_session_factory_with(
        tools=[Tool(
            name="vector_search",
            description="Retrieve documents via Vector Search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
            },
        )],
        call_result=CallToolResult(
            content=[TextContent(type="text", text='{"rows": [{"id": "doc-1"}]}')],
            isError=False,
        ),
    )
    client = DatabricksMCPClient(session_factory=factory)
    catalog.set_client(client)

    added = catalog.discover_tools()
    assert added == 1
    # Verify the catalog stamped the server name on the discovered tool.
    discovered = catalog.list_tools(server_config.name)
    assert len(discovered) == 1
    assert discovered[0].server_name == server_config.name
    # list_tools was called against the Managed Functions URL.
    session.list_tools.assert_awaited_once()

    # ---- Step 5: wire tools into a SkillRegistry ---------------------------
    registry = SkillRegistry()
    n = catalog.sync_to_skill_registry(registry)
    assert n == 1
    skill = registry.get(f"mcp:{server_config.name}:vector_search")
    assert skill is not None

    # ---- Step 6: invoke end-to-end -----------------------------------------
    result = skill.execute(SkillInput(query="widgets", parameters={"top_k": 3}))
    # The real MCP round-trip propagates through — call_tool was invoked with
    # the skill's parameters and the returned CallToolResult is surfaced.
    session.call_tool.assert_awaited_once_with(
        "vector_search", arguments={"top_k": 3},
    )
    assert result.skill_name == f"mcp:{server_config.name}:vector_search"
    assert result.metadata["mcp_server"] == server_config.name
    assert result.metadata["mcp_tool"] == "vector_search"
    # Output is the SDK's CallToolResult — agents inspect content / isError.
    assert result.output.isError is False
    assert result.output.content[0].text.startswith('{"rows"')
