import base64
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
        self.db_password = os.environ.get("LAKEBASE_DB_PASSWORD")
        self.db_user = self._resolve_db_user()
        self._ensure_table()

    def _resolve_db_user(self) -> str:
        """
        Resolve DB username for OAuth-based Lakebase auth.

        Priority:
        1) LAKEBASE_DB_USER override
        2) OAuth client_id (Databricks Apps / service principal runtime)
        3) current user name (interactive local/dev runtime)
        """
        explicit_user = os.environ.get("LAKEBASE_DB_USER")
        if explicit_user:
            return explicit_user
        client_id = getattr(self.workspace.config, "client_id", None)
        if client_id:
            return client_id
        return self.workspace.current_user.me().user_name

    def _token(self) -> str:
        return self.workspace.postgres.generate_database_credential(
            endpoint=self.endpoint_resource
        ).token

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

    def _connect(self) -> psycopg.Connection:
        if self.db_password:
            return psycopg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.db_user,
                password=self.db_password,
                sslmode="require",
            )

        token = self._token()
        token_subject = self._subject_from_token(token)
        explicit_user = os.environ.get("LAKEBASE_DB_USER")
        candidates: list[str] = []
        for candidate in [explicit_user, token_subject, self.db_user]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if os.environ.get("LAKEBASE_DEBUG_AUTH") == "1":
            print(
                "Lakebase auth debug:",
                {
                    "token_subject": token_subject,
                    "candidate_users": candidates,
                    "endpoint_resource": self.endpoint_resource,
                    "host": self.host,
                },
                flush=True,
            )

        last_error: Exception | None = None
        for user in candidates:
            try:
                if os.environ.get("LAKEBASE_DEBUG_AUTH") == "1":
                    print(f"Lakebase auth attempt user={user}", flush=True)
                return psycopg.connect(
                    host=self.host,
                    dbname=self.dbname,
                    user=user,
                    password=token,
                    sslmode="require",
                )
            except psycopg.OperationalError as exc:
                if os.environ.get("LAKEBASE_DEBUG_AUTH") == "1":
                    print(f"Lakebase auth failed user={user}: {exc}", flush=True)
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Lakebase DB user candidates were resolved")

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
        except Exception as exc:
            return False, f"Lakebase error: {exc}"
