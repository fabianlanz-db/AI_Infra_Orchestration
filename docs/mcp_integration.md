# MCP Integration

This framework lets any agent runtime (DSPy, LangGraph, external Pattern A
agents, custom) discover and invoke MCP tools on Databricks MCP servers
through a single governed path. Databricks Unity AI Gateway controls
access via Unity Catalog permissions, UC connections for external
credentials, and centralized audit.

References: [Databricks MCP docs](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/),
[AI Gateway](https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/),
[Connect non-Databricks clients](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/connect-external-services).

## Architecture

```
  agent runtime
       │
       ▼
  framework.mcp_client.DatabricksMCPClient        (uses the official `mcp` SDK)
       │
       │  MCP Streamable HTTP + Bearer auth
       ▼
  Databricks MCP server (managed / external / custom)
       │
       │  (for managed UC Functions)
       ▼
  Unity Catalog Functions  ← published via framework.uc_function_publisher
```

## Three server types

| Type     | URL pattern                                             | Auth supported      |
|----------|---------------------------------------------------------|---------------------|
| managed  | `<host>/api/2.0/mcp/functions/{catalog}/{schema}`       | OAuth U2M / M2M, PAT |
| managed  | `<host>/api/2.0/mcp/vector-search/{catalog}/{schema}`   | OAuth U2M / M2M, PAT |
| managed  | `<host>/api/2.0/mcp/genie/{genie_space_id}`             | OAuth U2M / M2M, PAT |
| external | URL from a Unity Catalog managed connection             | OAuth U2M / M2M, PAT |
| custom   | Databricks App URL (`<app>.cloud.databricksapps.com/mcp`) | **OAuth only** (no PAT) |

## Installation

Install with the `mcp` extra:

```bash
pip install -e ".[mcp]"
```

## Quickstart — in-workspace agent

```python
from databricks.sdk import WorkspaceClient
from framework.mcp_auth import WorkspaceClientAuth
from framework.mcp_client import DatabricksMCPClient
from framework.mcp_catalog_utils import MCPCatalogClient
from framework.mcp_servers import managed_functions_server

wc = WorkspaceClient()
auth = WorkspaceClientAuth(wc)            # auto-detects U2M / M2M / PAT
client = DatabricksMCPClient(auth=auth)

catalog = MCPCatalogClient()
catalog.set_client(client)
catalog.register_server(
    managed_functions_server("main", "agents", wc.config.host),
)
catalog.discover_tools()                  # populates via MCP list_tools

# Tools are now invokable as framework skills
from framework.skill_registry import SkillRegistry
registry = SkillRegistry()
catalog.sync_to_skill_registry(registry)
```

## Quickstart — external (Pattern A) agent

External agents hold service principal credentials and reach Databricks
over the network:

```python
# Environment: DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, DATABRICKS_HOST
from framework.mcp_auth import auto_select_auth
from framework.mcp_client import DatabricksMCPClient

auth = auto_select_auth()                 # picks SP M2M from env
client = DatabricksMCPClient(auth=auth)
# ... same catalog / discover / invoke flow as above
```

## Auth strategies

| Class                  | Mode tag | When to use                                              |
|------------------------|----------|----------------------------------------------------------|
| `WorkspaceClientAuth`  | auto     | Any time you have a `WorkspaceClient`; refresh is handled |
| `PATAuth`              | `pat`    | Dev/test; **rejected** by custom MCP                     |
| Pre-built session      | `byo`    | Advanced: pass `session_factory=...` to `DatabricksMCPClient` |

`ensure_auth_compatible_with_server` runs at registration time — if you
attach a PAT-backed client to a custom MCP server, it fails loudly
there rather than at the first tool invocation.

## Sync vs async

`DatabricksMCPClient` exposes both:

- `invoke_tool` / `list_tools` — sync, backed by `asyncio.run`. **Raises
  `RuntimeError` if called from inside an active event loop.**
- `ainvoke_tool` / `alist_tools` — async, use these inside any async
  runtime (FastAPI handlers, async adapters, etc.).

## Tracing

Every invocation emits `mcp.*` trace tags — see
[`observability_tags.md`](./observability_tags.md) for the schema. Pair
with `set_agent_tags(AgentContext(...))` at the start of your agent turn
so ACP can attribute the MCP call to a specific agent identity.

## Limitations

- There is no documented API to enumerate all MCP servers in a
  workspace. Callers must register the servers they want exposed
  (`managed_functions_server`, `register_server`). This is tracked as
  Q10 in the design doc and will be revisited when Databricks ships
  such an API.
- Stdio-based MCP (local `.cursor/mcp.json` style) is out of scope for
  governed invocation — the framework's `sync_from_mcp_json` still
  imports those entries for visibility, but they cannot be invoked
  through `DatabricksMCPClient`.
- Custom MCP does not support PAT; use OAuth.
