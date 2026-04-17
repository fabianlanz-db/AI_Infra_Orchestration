from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import psycopg
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# OAuth tokens from Databricks postgres typically live ~1h; refresh well before.
_TOKEN_TTL_SECONDS = 50 * 60


@dataclass
class MemoryWriteResult:
    event_id: int
    latency_ms: int


class LakebaseMemoryStore:
    """Reusable Lakebase utility with OAuth token generation and memory helpers.

    Exposes a public ``connection()`` context manager so adapters (e.g. the
    LangGraph checkpointer) can run raw SQL without reaching into private
    methods. OAuth tokens are cached for ``_TOKEN_TTL_SECONDS`` and the
    resolved DB user is memoized to avoid repeated candidate-probing.
    """

    def __init__(self) -> None:
        self.workspace = WorkspaceClient()
        self.endpoint_resource = os.environ.get("LAKEBASE_ENDPOINT_RESOURCE", "")
        if not self.endpoint_resource:
            raise ValueError("LAKEBASE_ENDPOINT_RESOURCE env var is required")
        self.host = os.environ.get("LAKEBASE_HOST", "")
        if not self.host:
            raise ValueError("LAKEBASE_HOST env var is required")
        self.dbname = os.environ.get("LAKEBASE_DB_NAME", "databricks_postgres")
        self.db_password = os.environ.get("LAKEBASE_DB_PASSWORD") or None
        self.db_user = self._resolve_db_user()
        self._cached_token: str | None = None
        self._cached_token_at: float = 0.0
        self._resolved_user: str | None = None
        self._table_ready = False

    def _resolve_db_user(self) -> str:
        explicit_user = os.environ.get("LAKEBASE_DB_USER")
        if explicit_user:
            return explicit_user
        client_id = getattr(self.workspace.config, "client_id", None)
        if client_id:
            return client_id
        return self.workspace.current_user.me().user_name

    def _token(self) -> str:
        now = time.time()
        if self._cached_token and (now - self._cached_token_at) < _TOKEN_TTL_SECONDS:
            return self._cached_token
        token = self.workspace.postgres.generate_database_credential(
            endpoint=self.endpoint_resource
        ).token
        self._cached_token = token
        self._cached_token_at = now
        return token

    def _subject_from_token(self, token: str) -> str | None:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload = parts[1]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
            claims = json.loads(decoded)
            return claims.get("sub")
        except Exception:
            return None

    def _open_connection(self) -> psycopg.Connection:
        if self.db_password:
            return psycopg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.db_user,
                password=self.db_password,
                sslmode="require",
            )

        token = self._token()

        if self._resolved_user:
            return psycopg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self._resolved_user,
                password=token,
                sslmode="require",
            )

        token_subject = self._subject_from_token(token)
        candidates: list[str] = []
        for candidate in [os.environ.get("LAKEBASE_DB_USER"), token_subject, self.db_user]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        logger.debug(
            "Lakebase auth: probing %d candidate(s) for endpoint=%s host=%s",
            len(candidates), self.endpoint_resource, self.host,
        )

        last_error: Exception | None = None
        for user in candidates:
            try:
                conn = psycopg.connect(
                    host=self.host,
                    dbname=self.dbname,
                    user=user,
                    password=token,
                    sslmode="require",
                )
                self._resolved_user = user
                return conn
            except psycopg.OperationalError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Lakebase DB user candidates were resolved")

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Public context manager yielding a connected psycopg connection.

        Ensures the session_memory table exists on first use.
        """
        conn = self._open_connection()
        try:
            if not self._table_ready:
                with conn.cursor() as cur:
                    cur.execute(_SESSION_MEMORY_DDL)
                    conn.commit()
                self._table_ready = True
            yield conn
        finally:
            conn.close()

    def write(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryWriteResult:
        start = time.perf_counter()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_memory (session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, role, content, json.dumps(metadata or {})),
            )
            event_id = cur.fetchone()[0]
            conn.commit()
        return MemoryWriteResult(event_id=event_id, latency_ms=int((time.perf_counter() - start) * 1000))

    def read(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, role, content, metadata, created_at
                FROM session_memory
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "metadata": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]

    def health(self) -> tuple[bool, str]:
        try:
            _ = self._token()
            return True, "Lakebase endpoint reachable"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"Lakebase error: {exc}"

    def write_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        assistant_metadata: dict[str, Any] | None = None,
    ) -> dict[str, MemoryWriteResult]:
        """Convenience hook for external APIs to persist one full turn."""
        user_result = self.write(
            session_id=session_id,
            role="user",
            content=user_message,
            metadata={"source": "external_agent"},
        )
        assistant_result = self.write(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            metadata=assistant_metadata or {},
        )
        return {"user": user_result, "assistant": assistant_result}

    def build_external_memory_payload(self, session_id: str, limit: int = 20) -> dict[str, Any]:
        """External API hook: export session memory in a stable payload shape."""
        events = self.read(session_id=session_id, limit=limit)
        return {
            "session_id": session_id,
            "event_count": len(events),
            "events": events,
        }


_SESSION_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS session_memory (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
