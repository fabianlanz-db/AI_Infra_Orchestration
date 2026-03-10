import json
import os
import time
from dataclasses import dataclass
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient


@dataclass
class MemoryWriteResult:
    event_id: int
    latency_ms: int


class LakebaseMemoryStore:
    """Reusable Lakebase utility with OAuth token generation and memory helpers."""

    def __init__(self) -> None:
        self.workspace = WorkspaceClient()
        self.endpoint_resource = os.environ.get(
            "LAKEBASE_ENDPOINT_RESOURCE",
            "projects/asml-external-agent-db/branches/production/endpoints/primary",
        )
        self.host = os.environ.get(
            "LAKEBASE_HOST",
            "ep-snowy-dawn-e1ff033t.database.eastus2.azuredatabricks.net",
        )
        self.dbname = os.environ.get("LAKEBASE_DB_NAME", "databricks_postgres")
        self._ensure_table()

    def _token(self) -> str:
        return self.workspace.postgres.generate_database_credential(
            endpoint=self.endpoint_resource
        ).token

    def _connect(self) -> psycopg.Connection:
        user = self.workspace.current_user.me().user_name
        return psycopg.connect(
            host=self.host,
            dbname=self.dbname,
            user=user,
            password=self._token(),
            sslmode="require",
        )

    def _ensure_table(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS session_memory (
          id BIGSERIAL PRIMARY KEY,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          metadata JSONB,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def write(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryWriteResult:
        start = time.perf_counter()
        with self._connect() as conn, conn.cursor() as cur:
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
        with self._connect() as conn, conn.cursor() as cur:
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
        """
        Convenience hook for external APIs to persist one full turn.

        Writes user message first, then assistant output.
        """
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
        """
        External API hook: export session memory in a stable payload shape.
        """
        events = self.read(session_id=session_id, limit=limit)
        return {
            "session_id": session_id,
            "event_count": len(events),
            "events": events,
        }
