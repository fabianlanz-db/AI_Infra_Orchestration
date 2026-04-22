# Observability Tag Schema

Stable contract for MLflow trace tags emitted by the framework. Downstream
consumers (Databricks Agent Control Plane, custom dashboards, evaluation
jobs) can rely on these keys and values being consistent across versions.

All tags are attached at **trace level** via `mlflow.update_current_trace`,
not span level — ACP and similar tools query at trace granularity.

Tagging is **best-effort**: when no active trace is present or the MLflow
backend is unreachable, tag helpers log at `DEBUG` and return silently.
Tagging will not block business logic.

## Namespaces

| Namespace  | Purpose                                              |
|------------|------------------------------------------------------|
| `agent.*`  | Identity of the caller (in-workspace or external)    |
| `routing.*`| Router decision metadata                             |
| `skill.*`  | Skill execution metadata                             |
| `mcp.*`    | MCP tool invocation metadata                         |

## `agent.*` — caller identity

Emitted by `set_agent_tags(AgentContext)` and `with_agent_context(...)`.

| Key               | Type     | Values / notes                                                  |
|-------------------|----------|-----------------------------------------------------------------|
| `agent.id`        | string   | Stable identifier chosen by the caller                          |
| `agent.origin`    | string   | `internal` (in-workspace) or `external` (Pattern A)             |
| `agent.framework` | string   | `dspy` / `langgraph` / `openapi` / `custom` / ...               |
| `agent.version`   | string   | Semver-ish; defaults to `0.0.0`                                 |
| `agent.principal` | string   | User or service principal; omitted if unknown                   |

Propagation across service boundaries uses these HTTP headers:

`x-agent-id` · `x-agent-origin` · `x-agent-framework` · `x-agent-version` · `x-agent-principal`

alongside the existing `x-mlflow-trace-id`.

## `routing.*` — router decisions

Emitted by each router tier (`RuleBasedRouter`, `LexicalRouter`,
`EmbeddingRouter`, `LLMRouter`). `CompositeRouter` re-tags with the
winning tier's name so the trace reflects composite-adjusted latency.

| Key                    | Type    | Values / notes                                             |
|------------------------|---------|------------------------------------------------------------|
| `routing.tier`         | string  | `rule` / `lexical` / `embedding` / `llm` / custom          |
| `routing.chosen_skill` | string  | Skill name dispatched to; empty if no decision             |
| `routing.confidence`   | float   | `[0.0, 1.0]`, rounded to 4 decimals                        |
| `routing.latency_ms`   | int     | From composite if cascade; from tier otherwise             |
| `routing.alternatives` | string  | Comma-separated runner-up skill names                      |
| `routing.rationale`    | string  | Human-readable; truncated to 200 chars                     |

## `skill.*` — skill execution

Emitted by `run_routed_turn` after successful skill execution.

| Key                 | Type   | Values / notes                                         |
|---------------------|--------|--------------------------------------------------------|
| `skill.name`        | string | Matches `SkillDefinition.name`                         |
| `skill.source`      | string | `local` / `unity_catalog` / `mcp`                      |
| `skill.version`     | string | From `SkillDefinition.version`                         |
| `skill.latency_ms`  | int    | From `SkillResult.latency_ms`                          |

## `mcp.*` — MCP tool invocations

Emitted by `DatabricksMCPClient.ainvoke_tool` on every MCP tool call.

| Key               | Type   | Values / notes                                                       |
|-------------------|--------|----------------------------------------------------------------------|
| `mcp.server`      | string | Logical server name as registered in `MCPCatalogClient`             |
| `mcp.server_type` | string | `managed` / `external` / `custom`                                    |
| `mcp.server_url`  | string | Actual HTTPS URL invoked (useful for pivoting by endpoint)           |
| `mcp.tool`        | string | Tool name on the server                                              |
| `mcp.latency_ms`  | int    | Wall-clock from session-enter through call_tool return               |
| `mcp.auth_mode`   | string | `u2m` / `m2m` / `pat` / `byo`                                        |

## Compatibility guarantees

Additions are backward-compatible. **Removing** or **renaming** a key is a
breaking change and will be called out in release notes.

## Relation to ACP

Databricks Agent Control Plane reads MLflow experiments from
`system.mlflow.runs_latest` and related system tables. Traces with these
tags attached can be filtered and aggregated in ACP dashboards to:

- Attribute cost and latency to specific agents (`agent.id`, `agent.framework`)
- Compare router-tier hit rates (`routing.tier`, `routing.confidence`)
- Audit MCP tool usage per server and auth mode (`mcp.*`)
