# Lakebase Session Memory

Persist conversation/session state in Databricks Lakebase (Postgres)
with OAuth-first authentication and optional password fallback.

## Use when

- Storing or retrieving conversation memory in Lakebase.
- Debugging Lakebase authentication failures in an app or runtime.
- Producing external API memory payloads for non-Databricks agents.

## Primary utilities

`framework/lakebase_utils.py`

- `LakebaseMemoryStore.write(...)` — low-level single event write.
- `LakebaseMemoryStore.read(...)` — read recent events for a session.
- `LakebaseMemoryStore.write_exchange(...)` — atomic user+assistant pair.
- `LakebaseMemoryStore.build_external_memory_payload(...)` — stable
  payload for external agents.
- `LakebaseMemoryStore.health()` — connectivity check.

## Authentication

1. **OAuth-first (recommended):** uses `generate_database_credential(...)`
   to produce a short-lived token; resolves candidate DB users from
   `LAKEBASE_DB_USER`, the token subject, and the runtime identity.
2. **Password fallback (optional):** `LAKEBASE_DB_PASSWORD` with
   `LAKEBASE_DB_USER` — useful for local dev only.
3. **Auth debug:** set `LAKEBASE_DEBUG_AUTH=1` temporarily if
   authentication appears misconfigured.

## Workflow

1. Instantiate `LakebaseMemoryStore()`.
2. Run `health()` for quick connectivity checks.
3. Persist one turn with `write_exchange(...)` when you have both
   user and assistant messages at hand.
4. Expose memory to external agents with
   `build_external_memory_payload(...)`.

## Notes

- Never commit plaintext secrets. Prefer OAuth or secret-scope-backed
  env injection.
- The OAuth token is refreshed proactively at the 50-minute mark of its
  ~1h lifetime so long-running sessions don't hiccup.
