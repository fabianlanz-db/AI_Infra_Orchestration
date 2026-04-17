"""
Skill registry for agent-agnostic tool discovery and execution.

Provides the ``SkillClient`` protocol, typed I/O dataclasses, and a
``SkillRegistry`` that supports keyword-overlap discovery out of the box.
For true semantic discovery, inject an embedder via
``SkillRegistry.configure_embedder()`` — when configured, ``discover()``
switches to cosine similarity over embedding vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from framework._text_utils import extract_terms

Embedder = Callable[[list[str]], list[list[float]]]


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

    Register skills via ``register()`` and discover them via ``discover()``.
    By default, discovery uses keyword overlap. Call
    ``configure_embedder(embedder)`` once at startup to upgrade discovery to
    true semantic cosine similarity over embedding vectors.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillClient] = {}
        self._embedder: Embedder | None = None
        self._embedding_cache: dict[str, list[float]] = {}

    def configure_embedder(self, embedder: Embedder) -> None:
        """Enable semantic ``discover()`` using the given embedder.

        The embedder must map a batch of texts to vectors. Skill embeddings
        are cached lazily keyed on ``name + description``; call
        :meth:`invalidate_embeddings` if a skill's description changes.
        """
        self._embedder = embedder
        self._embedding_cache.clear()

    def invalidate_embeddings(self) -> None:
        self._embedding_cache.clear()

    def register(self, skill: SkillClient) -> None:
        """Add a skill to the registry."""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill by name. No-op if not found."""
        self._skills.pop(name, None)
        self._embedding_cache.pop(name, None)

    def get(self, name: str) -> SkillClient | None:
        """Lookup a skill by exact name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return definitions of all registered skills."""
        return [s.definition for s in self._skills.values()]

    def discover(self, query: str, top_k: int = 5) -> list[SkillDefinition]:
        """Find skills matching a query.

        Uses cosine similarity if an embedder was configured via
        :meth:`configure_embedder`, otherwise falls back to keyword overlap
        scoring. Returns up to ``top_k`` definitions sorted by relevance.
        """
        if self._embedder is None:
            scored_kw: list[tuple[float, SkillDefinition]] = [
                (keyword_score(query, s.definition), s.definition)
                for s in self._skills.values()
            ]
            scored_kw.sort(key=lambda pair: pair[0], reverse=True)
            return [defn for score, defn in scored_kw[:top_k] if score > 0]

        return self._semantic_discover(query, top_k)

    def _semantic_discover(self, query: str, top_k: int) -> list[SkillDefinition]:
        assert self._embedder is not None
        defs = [s.definition for s in self._skills.values()]
        missing = [d for d in defs if d.name not in self._embedding_cache]
        if missing:
            texts = [f"{d.name}. {d.description}. tags: {', '.join(d.tags)}" for d in missing]
            vectors = self._embedder(texts)
            for d, v in zip(missing, vectors):
                self._embedding_cache[d.name] = v
        [qvec] = self._embedder([query])
        scored = [(self._cosine(qvec, self._embedding_cache[d.name]), d) for d in defs]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [defn for score, defn in scored[:top_k] if score > 0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

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
