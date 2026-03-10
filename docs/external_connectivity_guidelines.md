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

## 8) Framework Hooks for External Model APIs

Use these framework hooks to connect a non-Databricks model backend while
continuing to use Databricks retrieval, memory, and tracing:

### Module responsibilities (consolidated)

- `framework/external_model_hooks.py` is the canonical external integration module:
  - external-model adapter contract usage
  - OpenAPI/HTTP adapter implementation (`OpenApiModelClient`)
  - end-to-end orchestration (`run_external_agent_turn(...)`)
- `framework/openapi_model_adapter.py` is a compatibility import shim only.

- `framework/fm_agent_utils.py`
  - Implement `ExternalModelClient` (adapter interface).
  - Call `generate_with_external_client(...)` for provider-neutral generation.
- `framework/vector_search_utils.py`
  - Use `build_external_retrieval_payload(...)` to build a stable payload
    (`rows`, `context_block`, latency) for external model requests.
- `framework/lakebase_utils.py`
  - Use `write_exchange(...)` to persist user + assistant turn atomically.
  - Use `build_external_memory_payload(...)` to feed conversation history.
- `framework/mlflow_tracing_utils.py`
  - Use `build_trace_context_headers(...)` and
    `extract_trace_context_headers(...)` to propagate correlation IDs.
- `framework/external_model_hooks.py`
  - Use `OpenApiModelClient` for generic OpenAPI/HTTP model endpoints.
  - Configure `response_text_path` (for example `choices.0.message.content`)
    to map provider-specific response schemas.
  - Use `run_external_agent_turn(...)` as end-to-end reference orchestration.

### Minimal external adapter skeleton

```python
import time
from framework.fm_agent_utils import ExternalModelClient, ExternalModelRequest, FmResponse

class MyExternalApiClient(ExternalModelClient):
    def generate(self, request: ExternalModelRequest) -> FmResponse:
        start = time.perf_counter()
        # call your external HTTP/gRPC model API here
        text = "model output"
        return FmResponse(
            text=text,
            latency_ms=int((time.perf_counter() - start) * 1000),
            model="my-external-model",
        )

    def health(self) -> tuple[bool, str]:
        return True, "External model API reachable"
```

### OpenAPI adapter example (drop-in)

```python
from framework.external_model_hooks import OpenApiModelClient, OpenApiModelConfig, run_external_agent_turn

external_model = OpenApiModelClient(
    OpenApiModelConfig(
        inference_url="https://my-model.company.com/v1/chat/completions",
        headers={"Authorization": "Bearer <token>"},
        default_model="my-company-model",
        response_text_path="choices.0.message.content",
        health_url="https://my-model.company.com/health",
    )
)

result = run_external_agent_turn(
    query="How do we triage critical instability alerts?",
    session_id="sess-123",
    external_model_client=external_model,
    top_k=5,
)
print(result.response.text)
```
