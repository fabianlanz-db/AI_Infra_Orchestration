# External Model Integration

Wire any external (non-Databricks) model API into this framework's
orchestration flow. Applicable when you're hosting the LLM outside
Databricks — a vendor API, a self-hosted endpoint, or any HTTP/OpenAPI
surface — but still want Databricks Vector Search retrieval, Lakebase
memory, and MLflow tracing on the same turn.

## Use when

- Connecting a third-party LLM provider or a self-hosted model endpoint.
- Building on top of an OpenAPI-compatible chat completions API.
- Running one full agent turn with retrieval + generation + memory
  persistence end-to-end.

## Primary utilities

`framework/external_model_hooks.py`

- `ExternalModelClient` — adapter contract (Protocol) any implementation
  must satisfy.
- `OpenApiModelClient`, `OpenApiModelConfig` — reference implementation
  for HTTP/OpenAPI-compatible endpoints (OpenAI-shaped by default;
  override `build_request_body()` for other schemas).
- `generate_with_external_client(...)` — one-shot generation helper.
- `run_external_agent_turn(...)` — end-to-end orchestrator (retrieval +
  generation + memory + trace headers).

## Workflow

1. Implement or instantiate an `ExternalModelClient`. `OpenApiModelClient`
   covers most OpenAPI-style endpoints without subclassing.
2. Configure via `OpenApiModelConfig`:
   - `inference_url` — full POST URL
   - `headers` — auth and content headers
   - `response_text_path` — dotted path to the generated text
     (e.g. `choices.0.message.content`)
3. Call `run_external_agent_turn(...)` for the full flow. It performs:
   - Vector Search retrieval
   - External model generation
   - Lakebase memory write
   - Optional trace-header propagation
4. Inspect the returned `ExternalAgentTurnResult`:
   - `result.response.text` — the model output
   - `result.retrieval_payload` — context block used for generation
   - `result.memory_event_ids` — session memory IDs for both turns
   - `result.trace_headers` — propagate these to the next service hop

## Notes

- `framework/external_model_hooks.py` is the canonical implementation.
  `framework/openapi_model_adapter.py` is a compatibility re-export only.
- To tag the trace with agent identity, pass an `AgentContext` to
  `run_external_agent_turn`. See
  [`../observability_tags.md`](../observability_tags.md) for the schema.
- Prefer dependencies compatible with Databricks Serverless runtime.
