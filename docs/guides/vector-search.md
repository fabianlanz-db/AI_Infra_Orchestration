# Vector Search

Retrieve context from Databricks Vector Search indices through the
framework's typed client. Builds stable payloads suitable for in-process
agents and external model APIs.

## Use when

- Implementing retrieval against a Databricks Vector Search index.
- Producing external-agent payloads from retrieved chunks.
- Debugging Vector Search index connectivity or permissions.

## Primary utilities

`framework/vector_search_utils.py`

- `VectorSearchClient.retrieve(query_text, top_k)` — typed retrieval
  returning `RetrievalRow` instances.
- `VectorSearchClient.health()` — liveness + index reachability.
- `format_context_block(rows, top_k)` — render rows as a context block
  ready to inject into a prompt.
- `build_external_retrieval_payload(query, retrieval, top_k)` — stable
  payload for external model APIs.

## Workflow

1. Instantiate `VectorSearchClient` (optionally pass `index_name`).
2. Run `health()` for quick readiness checks when debugging.
3. Retrieve with `retrieve(query_text, top_k)`.
4. For external model APIs, use `build_external_retrieval_payload(...)`
   instead of hand-assembling payload shapes.
5. Feed `context_block` from the payload into your model's input context.

## Notes

- Default env var: `VS_INDEX_NAME`.
- Retrieval payload shape is intentionally stable — downstream consumers
  rely on it.
- The client uses `query_type="HYBRID"` by default (semantic + lexical).
