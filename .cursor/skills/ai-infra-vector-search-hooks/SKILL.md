---
name: ai-infra-vector-search-hooks
description: Uses this repository's Vector Search utilities to retrieve context and build stable payloads for internal or external agents. Use when implementing retrieval in the AI Infra Orchestration framework, debugging index access, or preparing model-ready context blocks from Databricks Vector Search.
---

# AI Infra Vector Search Hooks

Use this skill for retrieval logic in this repo.

## Use When

- User asks for retrieval setup with Databricks Vector Search.
- User needs external-agent payloads from retrieved chunks.
- User reports Vector Search index connectivity or permission issues.

## Primary Utilities

- `framework/vector_search_utils.py`
  - `VectorSearchClient.retrieve(query_text, top_k)`
  - `VectorSearchClient.health()`
  - `format_context_block(rows, top_k)`
  - `build_external_retrieval_payload(query, retrieval, top_k)`

## Required Workflow

1. Instantiate `VectorSearchClient` (optionally pass `index_name`).
2. Run `health()` for quick readiness checks when debugging.
3. Retrieve with `retrieve(query_text, top_k)`.
4. For external model APIs, use `build_external_retrieval_payload(...)` instead of custom payload shapes.
5. Use `context_block` from payload as model input context.

## Notes

- Default env var: `VS_INDEX_NAME`.
- Keep retrieval payload shape stable to avoid downstream breakage.
- Prefer `query_type="HYBRID"` behavior already set in utility.
