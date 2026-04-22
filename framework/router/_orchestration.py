"""Routing orchestration: ``run_routed_turn`` and ``RoutingJudge``.

Internal module. Consumers should import from ``framework.router``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from framework.mlflow_tracing_utils import (
    AgentContext,
    build_trace_context_headers,
    set_agent_tags,
    set_skill_tags,
    traced,
)
from framework.router._core import RouterClient, RoutingContext, RoutingDecision
from framework.skill_registry import SkillInput, SkillRegistry, SkillResult

if TYPE_CHECKING:
    from framework.judge_hooks import JudgeInput, JudgeVerdict


@dataclass
class RoutedTurnResult:
    """Full result from a routed turn (route + execute)."""

    routing: RoutingDecision
    result: SkillResult
    trace_headers: dict[str, str] = field(default_factory=dict)


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


@traced(name="router_run_routed_turn", span_type="CHAIN")
def run_routed_turn(
    query: str,
    session_id: str,
    router: RouterClient,
    registry: SkillRegistry,
    memory_store: Any | None = None,
    trace_id: str | None = None,
    agent_context: AgentContext | None = None,
) -> RoutedTurnResult:
    """Route query -> select skill -> execute -> optionally persist.

    Pass ``agent_context`` to tag the trace with agent identity so
    observability tools and dashboards can attribute the turn to its caller.
    """
    if agent_context is not None:
        set_agent_tags(agent_context)
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
        set_skill_tags(result, skill.definition)
    if memory_store and hasattr(memory_store, "write_exchange"):
        out = result.output
        text = str(out.get("text", out)) if isinstance(out, dict) else str(out)
        memory_store.write_exchange(
            session_id=session_id, user_message=query, assistant_message=text,
            assistant_metadata={"routed_skill": decision.skill_name, "confidence": decision.confidence},
        )
    trace_headers = build_trace_context_headers(trace_id) if trace_id else {}
    return RoutedTurnResult(routing=decision, result=result, trace_headers=trace_headers)
