# MLflow Tracing

Instrument agent flows with MLflow tracing, emit the framework's stable
tag schema, and propagate trace context across service hops. Pairs with
[`../observability_tags.md`](../observability_tags.md), which is the
reference contract for what ends up on each trace.

## Use when

- Adding MLflow tracing to an app or agent flow.
- Correlating trace IDs across external service calls.
- Diagnosing missing traces or validating trace visibility.
- Attributing a trace to a specific agent identity.

## Primary utilities

`framework/mlflow_tracing_utils.py`

**Span / configuration primitives**
- `configure_tracing(experiment_name, trace_destination)` — configure the
  tracking URI and destination; call once at app startup.
- `traced(name, span_type)` — decorator wrapping `mlflow.trace` for
  reusable span instrumentation.
- `verify_traces(max_results)` — quick runtime check that traces are
  landing where expected.

**Trace-id propagation (existing)**
- `build_trace_context_headers(trace_id)` — produce an `x-mlflow-trace-id`
  header for outbound calls.
- `extract_trace_context_headers(headers)` — normalize inbound headers
  into the same form for forwarding.

**Tag schema (Part B)**
- `AgentContext` — dataclass capturing agent identity
  (`id`, `origin`, `framework`, `version`, `principal`).
- `set_agent_tags(ctx)` / `set_routing_tags(decision, tier)` /
  `set_skill_tags(result, definition)` / `set_mcp_tags(...)` — emit
  the framework's stable tag namespaces onto the active trace.
- `with_agent_context(ctx)` — scope helper for DSPy, LangGraph, and
  other runtimes that don't own the HTTP boundary.
- `agent_context_from_headers(headers)` /
  `build_agent_context_headers(ctx)` — serialize agent identity
  across service hops (uses `x-agent-*` headers).
- `extract_agent_turn_context(headers)` — one-shot helper that pulls
  both the trace ID and the agent context from inbound headers.

## Workflow

1. Call `configure_tracing(...)` early in app startup.
2. Wrap key operations with `@traced(...)`.
3. Tag the trace with agent identity via `set_agent_tags(...)` or the
   `with_agent_context(...)` context manager.
4. For external service hops, propagate both the trace ID and agent
   identity via headers:
   - Outbound: merge
     `build_trace_context_headers(trace_id)` and
     `build_agent_context_headers(ctx)`.
   - Inbound: use `extract_agent_turn_context(headers)` to recover both.
5. Use `verify_traces(...)` for quick runtime confirmation.

## Notes

- Tagging is best-effort: when no active trace is present or MLflow is
  unreachable, tag helpers log at `DEBUG` and return silently — they
  never block business logic.
- Keep one clear root span per user prompt when possible.
- `mlflow.set_tracking_uri("databricks")` is handled by `configure_tracing`.
