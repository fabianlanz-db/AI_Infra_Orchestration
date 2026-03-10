# External Connectivity Guidelines (Lakebase + Vector Search)

This guide defines repeatable patterns for customer-managed external agents and applications that connect to Databricks Vector Search and Lakebase.

## 1) Architecture Patterns

### Pattern A (recommended): External App + Databricks APIs
- External runtime hosts agent logic.
- Vector retrieval is done through Databricks Vector Search APIs.
- Session/state is stored in Lakebase using short-lived OAuth credentials.

### Pattern B: Databricks App + FM endpoint + optional external tools
- Databricks App handles UI and orchestration.
- FM endpoint is used for generation.
- External services are called only for specific enterprise integrations.

## 2) Authentication and Identity

- Use Databricks SDK auth (`WorkspaceClient`) rather than hardcoding tokens.
- Prefer service principal auth for non-user workloads.
- Use least privilege access on UC objects, Vector Search indexes, and Lakebase endpoints.

## 3) Vector Search Access Pattern

### Recommended query mode
- `query_type="HYBRID"` for blended semantic + lexical retrieval.
- Keep `columns_to_sync` minimal but include all fields needed at inference time.

### Reliability guidance
- Add retries (exponential backoff) on transient failures.
- Bound request latency with client-side timeouts.
- Include retrieval metadata (`chunk_id`, score, source fields) in logs/traces.

## 4) Lakebase Access Pattern

### Token lifecycle
- Generate OAuth token per connection window:
  - `workspace.postgres.generate_database_credential(endpoint=...)`
- Tokens are short-lived (~1 hour). Add proactive refresh.

### Connection policy
- Always use `sslmode=require`.
- Implement retry for wake-up/reconnect paths.
- Use schema migration checks at startup (`CREATE TABLE IF NOT EXISTS`).

## 5) External Connectivity Hardening

- Use outbound allowlisting where available.
- Apply request timeouts on every network call.
- Use circuit-breaker behavior for dependent systems.
- Emit structured logs with correlation IDs.

## 6) Observability and MLflow Tracing

- Trace at these levels:
  - request entry
  - retrieval call
  - FM generation call
  - Lakebase read/write
- Verify traces in MLflow with `mlflow.search_traces()`.
- Use trace-derived datasets for `mlflow.genai.evaluate()` regression checks.
- For cross-service correlation, propagate `x-mlflow-trace-id` headers (see `build_trace_context_headers()` in `framework/mlflow_tracing_utils.py`).

## 7) Production Checklist

- [ ] FM endpoint latency/throughput validated.
- [ ] Vector index health and sync cadence validated.
- [ ] Lakebase token refresh + retry logic implemented.
- [ ] MLflow tracing enabled and validated.
- [ ] Evaluation baseline stored and comparable across releases.
