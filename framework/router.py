"""
Router protocol and reference implementations for intent-to-skill dispatch.

Provides RouterClient protocol, three reference routers (rule-based, semantic,
LLM-based), a composite cascading router, a RoutingJudge for evaluation, and
a ``run_routed_turn`` orchestrator that combines routing with skill execution.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from framework.judge_hooks import JudgeInput, JudgeVerdict

from framework.skill_registry import (
    SkillDefinition,
    SkillInput,
    SkillRegistry,
    SkillResult,
    keyword_score,
)


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


@dataclass
class RoutedTurnResult:
    """Full result from a routed turn (route + execute)."""

    routing: RoutingDecision
    result: SkillResult
    trace_headers: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class RouterClient(Protocol):
    """Adapter contract for routers — select the best skill for a query."""

    def route(self, query: str, context: RoutingContext) -> RoutingDecision: ...
    def health(self) -> tuple[bool, str]: ...


# -- Reference router implementations ----------------------------------------


class RuleBasedRouter:
    """Keyword/pattern router. Rules map regex patterns to skill names. First match wins."""

    def __init__(
        self,
        rules: list[tuple[str, str]] | None = None,
        default_skill: str = "generate",
    ) -> None:
        self._rules = [(re.compile(p, re.IGNORECASE), s) for p, s in (rules or [])]
        self._default = default_skill

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        for pattern, skill_name in self._rules:
            if pattern.search(query):
                return RoutingDecision(
                    skill_name=skill_name,
                    confidence=1.0,
                    rationale=f"Matched rule pattern: {pattern.pattern}",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
        return RoutingDecision(
            skill_name=self._default,
            confidence=0.5,
            rationale=f"No rule matched; defaulting to {self._default}",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def health(self) -> tuple[bool, str]:
        return True, f"RuleBasedRouter: {len(self._rules)} rule(s)"


class SemanticRouter:
    """Skill-description similarity router using keyword scoring."""

    def __init__(self, min_confidence: float = 0.1) -> None:
        self._min_confidence = min_confidence

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        skills = context.available_skills
        if not skills:
            return RoutingDecision(
                skill_name="", confidence=0.0, rationale="No skills available",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        scored = sorted(
            [(s, keyword_score(query, s)) for s in skills],
            key=lambda p: p[1], reverse=True,
        )
        best_skill, best_score = scored[0]
        if best_score == 0.0:
            return RoutingDecision(
                skill_name="", confidence=0.0,
                rationale="No keyword overlap with any skill",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        alternatives = [s.name for s, sc in scored[1:3] if sc > 0]
        conf = min(1.0, best_score)
        below = best_score < self._min_confidence
        rationale = (
            f"Below threshold ({best_score:.2f} < {self._min_confidence})" if below
            else f"Semantic match {best_score:.2f} for '{best_skill.name}'"
        )
        return RoutingDecision(
            skill_name=best_skill.name, confidence=conf, rationale=rationale,
            alternatives=alternatives, latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def health(self) -> tuple[bool, str]:
        return True, "SemanticRouter active"


class LLMRouter:
    """LLM-based router using an FM endpoint for intent classification."""

    def __init__(self, fm_client: Any) -> None:
        self._fm = fm_client

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        skills = context.available_skills
        skill_catalog = "\n".join(
            f"- {s.name}: {s.description} (tags: {', '.join(s.tags)})"
            for s in skills
        )
        system_prompt = (
            "You are a routing agent. Given a user query and available skills, "
            "select the best skill to handle the query. Respond with ONLY valid JSON: "
            '{"skill_name": "...", "confidence": 0.0-1.0, "rationale": "...", "parameters": {}}'
        )
        user_prompt = (
            f"Available skills:\n{skill_catalog}\n\nUser query: {query}\n\n"
            "Select the best skill and return JSON."
        )
        response = self._fm.generate(
            system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return self._parse_response(response.text, skills, latency_ms)

    def _parse_response(
        self, text: str, skills: list[SkillDefinition], latency_ms: int,
    ) -> RoutingDecision:
        try:
            cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
            data = json.loads(cleaned)
            skill_name = data.get("skill_name", "")
            valid_names = {s.name for s in skills}
            if skill_name not in valid_names and skills:
                skill_name = skills[0].name
            return RoutingDecision(
                skill_name=skill_name,
                confidence=float(data.get("confidence", 0.8)),
                rationale=data.get("rationale", "LLM routing decision"),
                parameters=data.get("parameters", {}),
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            fallback = skills[0].name if skills else ""
            return RoutingDecision(
                skill_name=fallback, confidence=0.3,
                rationale=f"LLM response parsing failed; falling back to {fallback}",
                latency_ms=latency_ms,
            )

    def health(self) -> tuple[bool, str]:
        return self._fm.health()


class CompositeRouter:
    """Cascading router: rules -> semantic -> LLM, stops at first confident match."""

    def __init__(
        self,
        routers: list[tuple[RouterClient, float]] | None = None,
    ) -> None:
        self._tiers: list[tuple[RouterClient, float]] = routers or []

    def add_tier(self, router: RouterClient, min_confidence: float = 0.7) -> None:
        """Add a router tier with a minimum confidence threshold."""
        self._tiers.append((router, min_confidence))

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        last_decision: RoutingDecision | None = None
        for router, threshold in self._tiers:
            last_decision = router.route(query, context)
            if last_decision.confidence >= threshold:
                last_decision.latency_ms = int((time.perf_counter() - start) * 1000)
                return last_decision
        total_ms = int((time.perf_counter() - start) * 1000)
        if last_decision:
            last_decision.latency_ms = total_ms
            last_decision.rationale = f"No tier met threshold; using last: {last_decision.rationale}"
            return last_decision
        return RoutingDecision(
            skill_name="", confidence=0.0,
            rationale="No router tiers configured", latency_ms=total_ms,
        )

    def health(self) -> tuple[bool, str]:
        return True, f"CompositeRouter: {len(self._tiers)} tier(s)"


# -- Routing evaluation --------------------------------------------------------

class RoutingJudge:
    """Judge evaluating routing accuracy. Satisfies JudgeClient protocol."""

    @property
    def name(self) -> str:
        return "routing_accuracy"

    def evaluate(self, input: JudgeInput) -> JudgeVerdict:
        from framework.judge_hooks import JudgeVerdict as _JV

        expected = input.expectations.get("expected_skill", "")
        actual = input.response
        passed = expected.lower() == actual.lower() if expected else True
        return _JV(
            judge_name=self.name, passed=passed,
            score=1.0 if passed else 0.0,
            rationale=(
                f"Routed to '{actual}', expected '{expected}'."
                if expected else "No expected skill specified."
            ),
        )


# -- Orchestration function ---------------------------------------------------


def run_routed_turn(
    query: str,
    session_id: str,
    router: RouterClient,
    registry: SkillRegistry,
    memory_store: Any | None = None,
    trace_id: str | None = None,
) -> RoutedTurnResult:
    """Route query -> select skill -> execute -> optionally persist."""
    context = RoutingContext(session_id=session_id, available_skills=registry.list_skills())
    decision = router.route(query, context)
    skill = registry.get(decision.skill_name)
    if not skill:
        result = SkillResult(
            output={"error": f"Skill '{decision.skill_name}' not found"},
            latency_ms=0, skill_name=decision.skill_name,
        )
    else:
        skill_input = SkillInput(
            query=query, parameters=decision.parameters,
            session_id=session_id, trace_id=trace_id,
        )
        result = skill.execute(skill_input)
    if memory_store and hasattr(memory_store, "write_exchange"):
        out = result.output
        text = str(out.get("text", out)) if isinstance(out, dict) else str(out)
        memory_store.write_exchange(
            session_id=session_id, user_message=query, assistant_message=text,
            assistant_metadata={"routed_skill": decision.skill_name, "confidence": decision.confidence},
        )
    trace_headers: dict[str, str] = {}
    if trace_id:
        from framework.mlflow_tracing_utils import build_trace_context_headers
        trace_headers = build_trace_context_headers(trace_id)
    return RoutedTurnResult(routing=decision, result=result, trace_headers=trace_headers)
