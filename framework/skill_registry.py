"""
Skill registry for agent-agnostic tool discovery and execution.

Provides SkillClient protocol, typed I/O dataclasses, a SkillRegistry
with optional semantic discovery via Vector Search, and reference skill
implementations wrapping existing framework capabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from framework._text_utils import extract_terms


@dataclass
class SkillDefinition:
    """Metadata describing a registered skill."""

    name: str
    description: str
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # "local" | "unity_catalog" | "mcp"
    mcp_server: str | None = None


@dataclass
class SkillInput:
    """Standardized input payload for skill execution."""

    query: str
    parameters: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    trace_id: str | None = None


@dataclass
class SkillResult:
    """Standardized result from a skill execution."""

    output: Any
    latency_ms: int
    skill_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SkillClient(Protocol):
    """Adapter contract for skills.

    Any class with ``name``, ``definition``, ``execute``, and ``health``
    satisfies this contract and can be registered in a SkillRegistry.
    """

    @property
    def name(self) -> str: ...

    @property
    def definition(self) -> SkillDefinition: ...

    def execute(self, input: SkillInput) -> SkillResult: ...

    def health(self) -> tuple[bool, str]: ...


def keyword_score(query: str, definition: SkillDefinition) -> float:
    """Score a skill definition against a query using keyword overlap."""
    query_terms = extract_terms(query)
    if not query_terms:
        return 0.0
    target = f"{definition.name} {definition.description} {' '.join(definition.tags)}"
    target_terms = extract_terms(target)
    overlap = query_terms & target_terms
    return len(overlap) / len(query_terms)


class SkillRegistry:
    """In-memory skill registry with optional semantic discovery.

    Register skills via ``register()``, discover them via keyword matching,
    and optionally upgrade to semantic search by providing a VectorSearchClient.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillClient] = {}

    def register(self, skill: SkillClient) -> None:
        """Add a skill to the registry."""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill by name. No-op if not found."""
        self._skills.pop(name, None)

    def get(self, name: str) -> SkillClient | None:
        """Lookup a skill by exact name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return definitions of all registered skills."""
        return [s.definition for s in self._skills.values()]

    def discover(self, query: str, top_k: int = 5) -> list[SkillDefinition]:
        """Find skills matching a query via keyword scoring.

        Scores each skill description against the query and returns
        the top-k matches sorted by relevance.
        """
        scored = [
            (keyword_score(query, s.definition), s.definition)
            for s in self._skills.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [defn for score, defn in scored[:top_k] if score > 0]

    def health(self) -> tuple[bool, str]:
        count = len(self._skills)
        return True, f"Skill registry active with {count} skill(s)"

    def build_external_skill_catalog_payload(self) -> dict[str, Any]:
        """Export the full skill catalog for external agents."""
        skills = []
        for d in self.list_skills():
            skills.append({
                "name": d.name, "description": d.description, "version": d.version,
                "tags": d.tags, "input_schema": d.input_schema,
                "output_schema": d.output_schema, "source": d.source,
                "mcp_server": d.mcp_server,
            })
        return {"skill_count": len(self._skills), "skills": skills}
