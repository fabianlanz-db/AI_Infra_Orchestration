"""
Reference skill implementations wrapping existing framework capabilities.

Each skill implements the SkillClient protocol from skill_registry.py
and delegates to a concrete Databricks client (VectorSearchClient,
LakebaseMemoryStore, FmAgentClient).
"""

from __future__ import annotations

import time

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.skill_registry import SkillDefinition, SkillInput, SkillResult
from framework.vector_search_utils import VectorSearchClient


class VectorSearchSkill:
    """Skill wrapping VectorSearchClient.retrieve()."""

    def __init__(self, vector_client: VectorSearchClient) -> None:
        self._client = vector_client

    @property
    def name(self) -> str:
        return "vector-search"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description="Retrieve relevant documents from Databricks Vector Search using hybrid semantic and lexical matching.",
            tags=["retrieval", "rag", "databricks", "search"],
            input_schema={"query": "str", "top_k": "int (default 5)"},
            output_schema={"rows": "list[list[Any]]", "retrieval_latency_ms": "int"},
        )

    def execute(self, input: SkillInput) -> SkillResult:
        top_k = input.parameters.get("top_k", 5)
        start = time.perf_counter()
        result = self._client.retrieve(input.query, top_k=top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SkillResult(
            output={"rows": result.rows, "retrieval_latency_ms": result.latency_ms},
            latency_ms=latency_ms,
            skill_name=self.name,
        )

    def health(self) -> tuple[bool, str]:
        return self._client.health()


class MemoryReadSkill:
    """Skill wrapping LakebaseMemoryStore.read()."""

    def __init__(self, memory_store: LakebaseMemoryStore) -> None:
        self._store = memory_store

    @property
    def name(self) -> str:
        return "memory-read"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description="Read conversation history from Databricks Lakebase session memory.",
            tags=["memory", "session", "databricks", "lakebase"],
            input_schema={"session_id": "str (required in parameters)", "limit": "int (default 20)"},
            output_schema={"events": "list[dict]", "event_count": "int"},
        )

    def execute(self, input: SkillInput) -> SkillResult:
        session_id = input.session_id or input.parameters.get("session_id", "")
        limit = input.parameters.get("limit", 20)
        start = time.perf_counter()
        events = self._store.read(session_id=session_id, limit=limit)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SkillResult(
            output={"events": events, "event_count": len(events)},
            latency_ms=latency_ms,
            skill_name=self.name,
        )

    def health(self) -> tuple[bool, str]:
        return self._store.health()


class MemoryWriteSkill:
    """Skill wrapping LakebaseMemoryStore.write()."""

    def __init__(self, memory_store: LakebaseMemoryStore) -> None:
        self._store = memory_store

    @property
    def name(self) -> str:
        return "memory-write"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description="Write a message to Databricks Lakebase session memory.",
            tags=["memory", "session", "databricks", "lakebase", "write"],
            input_schema={"session_id": "str", "role": "str", "content": "str"},
            output_schema={"event_id": "int"},
        )

    def execute(self, input: SkillInput) -> SkillResult:
        session_id = input.session_id or input.parameters.get("session_id", "")
        role = input.parameters.get("role", "user")
        content = input.query
        start = time.perf_counter()
        result = self._store.write(session_id=session_id, role=role, content=content)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SkillResult(
            output={"event_id": result.event_id},
            latency_ms=latency_ms,
            skill_name=self.name,
        )

    def health(self) -> tuple[bool, str]:
        return self._store.health()


class GenerateSkill:
    """Skill wrapping FmAgentClient.generate()."""

    def __init__(self, fm_client: FmAgentClient) -> None:
        self._client = fm_client

    @property
    def name(self) -> str:
        return "generate"

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description="Generate text using a Databricks FM serving endpoint (LLM inference).",
            tags=["generation", "llm", "databricks", "inference"],
            input_schema={"system_prompt": "str (optional)", "temperature": "float (default 0.2)"},
            output_schema={"text": "str", "model": "str"},
        )

    def execute(self, input: SkillInput) -> SkillResult:
        system_prompt = input.parameters.get(
            "system_prompt", "You are a helpful assistant."
        )
        temperature = input.parameters.get("temperature", 0.2)
        start = time.perf_counter()
        response = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=input.query,
            temperature=temperature,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SkillResult(
            output={"text": response.text, "model": response.model},
            latency_ms=latency_ms,
            skill_name=self.name,
        )

    def health(self) -> tuple[bool, str]:
        return self._client.health()
