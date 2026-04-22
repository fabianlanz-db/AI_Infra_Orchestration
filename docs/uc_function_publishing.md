# Publishing Skills as Unity Catalog Functions

Publish a framework skill as a UC Function via
`framework.uc_function_publisher`. The payoff: **any MCP-capable agent
automatically discovers your skill as a tool** through Databricks' managed
Functions MCP server at `<host>/api/2.0/mcp/functions/{catalog}/{schema}`.

This is the tightest coupling between the three tool surfaces — skills
(native), UC Functions (governed), MCP tools (discoverable). Publishing
once produces all three views.

## Deployment model — service mode

The generated UC Function body POSTs to a **Model Serving endpoint**
hosting the skill. The framework does **not** host skills; you do that
by deploying them behind a serving endpoint.

```
  MCP client ──▶ Managed Functions MCP ──▶ UC Function ──▶ Model Serving ──▶ Skill
                 (AI Gateway-governed)      (this module)   (your deployment)
```

## Quickstart

```python
from databricks.sdk import WorkspaceClient
from framework.reference_uc_bindings import vector_search_binding
from framework.uc_function_publisher import publish_skill

binding = vector_search_binding(
    catalog="main",
    schema="agents",
    serving_endpoint="https://<host>/serving-endpoints/my-vs-skill/invocations",
)
fq = publish_skill(binding, WorkspaceClient(), sql_warehouse_id="wh-abc123")
# fq = "main.agents.vector_search"
```

After publish, Databricks exposes the function through managed Functions
MCP automatically — a separate framework step is not required.

## Dry-run the DDL

`render_function_sql` is a pure function. Use it to lint the DDL before
any workspace-side execution:

```python
from framework.uc_function_publisher import render_function_sql
print(render_function_sql(binding))
```

## Building custom bindings

```python
from framework.uc_function_publisher import SkillUCBinding, UCColumn

binding = SkillUCBinding(
    skill_name="my-skill",
    catalog="main", schema="agents",
    function_name="my_tool",
    serving_endpoint="https://<host>/serving-endpoints/my-skill/invocations",
    input_columns=[
        UCColumn("query", "STRING", comment="User question"),
        UCColumn("top_k", "INT", default="5"),
    ],
    return_type="STRING",
    comment="Custom tool for my agent.",
    query_column="query",      # which column feeds SkillInput.query
    timeout_seconds=30,
)
```

Validation runs at render time: empty `input_columns`, missing
`query_column`, or duplicate column names raise `ValueError`.

## Skills omitted from reference bindings

`MemoryReadSkill` and `MemoryWriteSkill` are **intentionally not** in
`reference_uc_bindings.py`: session memory is too permissions-sensitive
to expose as a UC Function callable by arbitrary principals. Publish a
scoped variant only if you have UC grants that restrict it to the
caller's own session.

## Auth inside the UC Function body

The generated Python body reads `DATABRICKS_TOKEN` from the function's
execution environment and uses it as a Bearer credential against the
serving endpoint. In Databricks, UC Python Functions inherit the
invoker's identity for calls to governed services, so this works when
the invoker has Model Serving access.

For service-principal backends (the function always calls the endpoint
with the same SP regardless of caller), the recommended pattern is a
Databricks Secret scope referenced by the function body. The current
release does not parameterize this; fork the generator or open an issue
if you need it.

## Idempotency

`publish_skill` emits `CREATE OR REPLACE FUNCTION`, so re-publishing the
same binding is safe — it overwrites the existing definition.
`publish_bindings` is a thin batch wrapper over `publish_skill`.

## Relation to ACP

UC Functions published via this module show up in ACP's tool registry
view alongside Databricks-native UC functions. Invocations through the
Managed Functions MCP endpoint are billed/attributed to the invoker's
principal via `system.billing.usage` and `system.access.audit` — the
same path ACP already queries for cost and audit attribution.
