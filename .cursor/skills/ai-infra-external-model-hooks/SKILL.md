---
name: ai-infra-external-model-hooks
description: Integrates external model APIs with the AI Infra Orchestration framework using provider-neutral hooks, OpenAPI adapters, Vector Search retrieval context, Lakebase memory writes, and optional MLflow trace header propagation. Use when connecting non-Databricks model endpoints, OpenAPI model APIs, or external agent backends.
---

# AI Infra External Model Hooks

Use this skill to wire any external model API into this repo's orchestration flow.

## Use When

- User asks to connect an external LLM/model API.
- User mentions OpenAPI model adapters or provider-neutral model integration.
- You need to run one full turn with retrieval + generation + memory persistence.

## Primary Utilities

- `framework/external_model_hooks.py`
  - `ExternalModelClient` (protocol)
  - `OpenApiModelClient`, `OpenApiModelConfig`
  - `generate_with_external_client(...)`
  - `run_external_agent_turn(...)`
- `framework/openapi_model_adapter.py`
  - Compatibility re-export only.

## Required Workflow

1. Implement/instantiate an `ExternalModelClient`.
   - Prefer `OpenApiModelClient` for HTTP/OpenAPI-compatible endpoints.
2. Build config with `OpenApiModelConfig`.
   - Set `inference_url`, auth headers, `response_text_path`.
3. Call `run_external_agent_turn(...)` for end-to-end execution.
   - This automatically performs:
     - Vector Search retrieval
     - external model generation
     - Lakebase memory write
     - optional trace header return
4. Return/use:
   - `result.response.text`
   - `result.retrieval_payload`
   - `result.memory_event_ids`
   - `result.trace_headers`

## Notes

- Keep `framework/external_model_hooks.py` as canonical implementation.
- Do not add duplicate orchestration logic in `openapi_model_adapter.py`.
- Prefer Databricks Serverless-compatible dependencies and runtime behavior.
