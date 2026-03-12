---
name: ai-infra-lakebase-memory-hooks
description: Applies this repository's Lakebase memory utility patterns for chat/session persistence with OAuth-first authentication and optional password fallback. Use when implementing memory writes/reads, debugging Lakebase auth, or exporting stable memory payloads for external agent APIs.
---

# AI Infra Lakebase Memory Hooks

Use this skill for session memory persistence in Lakebase.

## Use When

- User asks to store/retrieve conversation memory in Lakebase.
- User sees Lakebase authentication failures in app/runtime.
- User wants external API memory payloads from this framework.

## Primary Utilities

- `framework/lakebase_utils.py`
  - `LakebaseMemoryStore.write(...)`
  - `LakebaseMemoryStore.read(...)`
  - `LakebaseMemoryStore.write_exchange(...)`
  - `LakebaseMemoryStore.build_external_memory_payload(...)`
  - `LakebaseMemoryStore.health()`

## Authentication Rules

1. OAuth-first:
   - uses `generate_database_credential(...)` token.
   - resolves candidate DB users from `LAKEBASE_DB_USER`, token subject, and runtime identity.
2. Optional fallback:
   - `LAKEBASE_DB_PASSWORD` with `LAKEBASE_DB_USER`.
3. Debug only when needed:
   - set `LAKEBASE_DEBUG_AUTH=1` temporarily.

## Required Workflow

1. Instantiate `LakebaseMemoryStore()`.
2. Run `health()` for quick connectivity checks.
3. Persist one turn with `write_exchange(...)` when possible.
4. Expose memory externally with `build_external_memory_payload(...)`.

## Notes

- Keep app behavior compatible with Databricks Serverless and latest DBR runtimes.
- Never commit new plaintext secrets; prefer OAuth or secret-backed env injection.
