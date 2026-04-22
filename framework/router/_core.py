"""Core router types: protocol, dataclasses, and shared tagging helpers.

Internal module. Consumers should import from ``framework.router``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from framework.mlflow_tracing_utils import set_routing_tags
from framework.skill_registry import SkillDefinition


@dataclass
class RoutingContext:
    """Context provided to a router for making routing decisions."""

    session_id: str | None = None
    available_skills: list[SkillDefinition] = field(default_factory=list)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    skill_name: str
    confidence: float  # 0.0 to 1.0
    rationale: str
    parameters: dict[str, Any] = field(default_factory=dict)
    alternatives: list[str] = field(default_factory=list)
    latency_ms: int = 0


@runtime_checkable
class RouterClient(Protocol):
    """Adapter contract for routers — select the best skill for a query.

    Implementations receive a ``RoutingContext`` whose fields are used by
    different routers to varying degrees:

    * ``available_skills`` — REQUIRED for semantic/lexical/LLM routers. Rule-
      based routers ignore it.
    * ``session_id`` — surfaced to routers that score by conversational state.
      Currently unused by the reference implementations but reserved.
    * ``conversation_history`` — consumed by routers that condition on prior
      turns (e.g. a stateful LLM router). Reference routers ignore it.
    * ``metadata`` — free-form dict for experimental signals (e.g. user tier,
      tenant flags). Reference routers ignore it.

    A conforming router MUST return a ``RoutingDecision`` with ``confidence``
    in ``[0.0, 1.0]``. Return ``skill_name=""`` with ``confidence=0.0`` if no
    decision can be made; a ``CompositeRouter`` will fall through to the next
    tier.
    """

    def route(self, query: str, context: RoutingContext) -> RoutingDecision: ...
    def health(self) -> tuple[bool, str]: ...


def tier_name_of(router: object) -> str:
    """Extract a tier name for tagging; falls back to the class name."""
    name = getattr(router, "tier_name", None)
    if isinstance(name, str) and name:
        return name
    return router.__class__.__name__.lower().replace("router", "") or "custom"


def tagged(decision: RoutingDecision, tier_name: str) -> RoutingDecision:
    """Tag the active trace with ``routing.*`` and return the decision.

    Used at every ``route()`` return site to collapse the three-line
    ``decision = ...; set_routing_tags(...); return decision`` pattern into
    a single ``return tagged(RoutingDecision(...), self.tier_name)``.
    """
    set_routing_tags(decision, tier_name)
    return decision
