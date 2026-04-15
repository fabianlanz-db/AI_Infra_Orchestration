"""
LangGraph adapter layer for Databricks infrastructure.

Provides adapter classes so LangGraph graphs can use Databricks Lakebase
for checkpointing, FM endpoints as chat models, and SkillRegistry skills
as LangChain tools.

LangGraph and LangChain are optional dependencies. All adapters are
guarded by import checks and raise clear errors if not installed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.skill_registry import SkillClient, SkillInput, SkillRegistry

try:
    from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401 - availability check

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

try:
    from langchain_core.tools import BaseTool  # noqa: F401 - used in skill_as_langchain_tool

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def _require_langgraph() -> None:
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is required for this adapter. "
            "Install it with: pip install langgraph"
        )


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain Core is required for this adapter. "
            "Install it with: pip install langchain-core"
        )


class LakebaseCheckpointer:
    """LangGraph checkpoint saver backed by Databricks Lakebase.

    Stores graph state as JSON in a ``graph_checkpoints`` table.
    Compatible with LangGraph's StateGraph checkpointing protocol.

    Usage::

        from framework.langgraph_adapter import LakebaseCheckpointer
        checkpointer = LakebaseCheckpointer()
        graph = StateGraph(...).compile(checkpointer=checkpointer)
    """

    TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS graph_checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL,
        parent_id TEXT,
        checkpoint JSONB NOT NULL,
        metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (thread_id, checkpoint_id)
    );
    """

    def __init__(self, memory_store: LakebaseMemoryStore | None = None) -> None:
        _require_langgraph()
        self._store = memory_store or LakebaseMemoryStore()
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Uses LakebaseMemoryStore._connect() directly for raw SQL access.
        # If LakebaseMemoryStore gains a public execute_sql(), migrate to that.
        with self._store._connect() as conn, conn.cursor() as cur:
            cur.execute(self.TABLE_DDL)
            conn.commit()

    def put(self, config: dict[str, Any], checkpoint: dict[str, Any],
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save a checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = checkpoint.get("id", str(time.time_ns()))
        parent_id = config.get("configurable", {}).get("checkpoint_id")
        with self._store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO graph_checkpoints (thread_id, checkpoint_id, parent_id, checkpoint, metadata)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (thread_id, checkpoint_id) DO UPDATE SET
                     checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata""",
                (thread_id, checkpoint_id, parent_id,
                 json.dumps(checkpoint), json.dumps(metadata or {})),
            )
            conn.commit()
        return {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Load the latest checkpoint for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        with self._store._connect() as conn, conn.cursor() as cur:
            if checkpoint_id:
                cur.execute(
                    "SELECT checkpoint, metadata FROM graph_checkpoints WHERE thread_id=%s AND checkpoint_id=%s",
                    (thread_id, checkpoint_id),
                )
            else:
                cur.execute(
                    "SELECT checkpoint, metadata FROM graph_checkpoints WHERE thread_id=%s ORDER BY created_at DESC LIMIT 1",
                    (thread_id,),
                )
            row = cur.fetchone()
        if not row:
            return None
        return {"checkpoint": row[0], "metadata": row[1]}

    def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """List recent checkpoints for a thread."""
        with self._store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT checkpoint_id, parent_id, metadata, created_at FROM graph_checkpoints WHERE thread_id=%s ORDER BY created_at DESC LIMIT %s",
                (thread_id, limit),
            )
            rows = cur.fetchall()
        return [
            {"checkpoint_id": r[0], "parent_id": r[1], "metadata": r[2],
             "created_at": r[3].isoformat() if r[3] else None}
            for r in rows
        ]

    def health(self) -> tuple[bool, str]:
        return self._store.health()


def skill_as_langchain_tool(skill: SkillClient) -> Any:
    """Convert a SkillClient to a LangChain-compatible tool.

    Usage::

        from framework.langgraph_adapter import skill_as_langchain_tool
        tools = [skill_as_langchain_tool(s) for s in registry.list_skills()]
    """
    _require_langchain()

    class _SkillTool(BaseTool):
        name: str = skill.name
        description: str = skill.definition.description

        def _run(self, query: str, **kwargs: Any) -> str:
            result = skill.execute(SkillInput(query=query, parameters=kwargs))
            out = result.output
            return str(out.get("text", out)) if isinstance(out, dict) else str(out)

    return _SkillTool()


class DatabricksChatModel:
    """LangChain chat model backed by a Databricks FM endpoint.

    Usage::

        from framework.langgraph_adapter import DatabricksChatModel
        llm = DatabricksChatModel()
    """

    def __init__(self, fm_client: FmAgentClient | None = None) -> None:
        _require_langchain()
        self._fm = fm_client or FmAgentClient()

    @property
    def model_name(self) -> str:
        return self._fm.endpoint_name

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Invoke the model with a message list."""
        system = ""
        user = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user = content
        response = self._fm.generate(
            system_prompt=system or "You are a helpful assistant.",
            user_prompt=user, temperature=kwargs.get("temperature", 0.2),
        )
        return {"content": response.text, "model": response.model, "latency_ms": response.latency_ms}

    def health(self) -> tuple[bool, str]:
        return self._fm.health()


def build_langgraph_tools(registry: SkillRegistry) -> list[Any]:
    """Convert all skills in a registry to LangChain tools."""
    _require_langchain()
    tools = []
    for defn in registry.list_skills():
        skill = registry.get(defn.name)
        if skill:
            tools.append(skill_as_langchain_tool(skill))
    return tools
