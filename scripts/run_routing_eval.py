"""
Evaluate routing accuracy against expected skill mappings.

Builds an eval dataset of queries with expected skill targets, routes
each query through a CompositeRouter, and scores routing accuracy
using the RoutingJudge and MLflow evaluation.

Usage:
    python scripts/run_routing_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.judge_hooks import JudgeInput
from framework.router import (
    CompositeRouter,
    RoutingContext,
    RoutingJudge,
    RuleBasedRouter,
    SemanticRouter,
)
from framework.skill_registry import SkillRegistry

EVAL_DATASET = [
    {"query": "search for turbine maintenance alerts", "expected_skill": "vector-search"},
    {"query": "find documents about cooling system failures", "expected_skill": "vector-search"},
    {"query": "retrieve knowledge base entries for vibration issues", "expected_skill": "vector-search"},
    {"query": "show me my conversation history", "expected_skill": "memory-read"},
    {"query": "what did we discuss in the last session", "expected_skill": "memory-read"},
    {"query": "remember what I asked about earlier", "expected_skill": "memory-read"},
    {"query": "summarize the maintenance report for unit 7", "expected_skill": "generate"},
    {"query": "explain the root cause of the thermal anomaly", "expected_skill": "generate"},
    {"query": "write a brief analysis of the sensor data trends", "expected_skill": "generate"},
    {"query": "generate a risk assessment for the compressor", "expected_skill": "generate"},
]


def _build_mock_registry() -> SkillRegistry:
    """Build a registry with stub skills for evaluation (no live backends)."""

    class _StubSkill:
        def __init__(self, name: str, desc: str, tags: list[str]) -> None:
            self._name = name
            self._desc = desc
            self._tags = tags

        @property
        def name(self) -> str:
            return self._name

        @property
        def definition(self):
            from framework.skill_registry import SkillDefinition
            return SkillDefinition(name=self._name, description=self._desc, tags=self._tags)

        def execute(self, inp):
            from framework.skill_registry import SkillResult
            return SkillResult(output={"stub": True}, latency_ms=0, skill_name=self._name)

        def health(self):
            return True, f"{self._name} stub"

    registry = SkillRegistry()
    registry.register(_StubSkill(
        "vector-search", "Retrieve documents from Databricks Vector Search using hybrid matching.",
        ["retrieval", "rag", "search", "find", "documents"],
    ))
    registry.register(_StubSkill(
        "memory-read", "Read conversation history from Lakebase session memory.",
        ["memory", "session", "history", "remember", "conversation"],
    ))
    registry.register(_StubSkill(
        "generate", "Generate text using a Databricks FM serving endpoint.",
        ["generation", "llm", "summarize", "explain", "write", "analyze"],
    ))
    return registry


def run_eval() -> dict:
    """Run routing evaluation and return results."""
    registry = _build_mock_registry()

    router = CompositeRouter()
    router.add_tier(RuleBasedRouter(rules=[
        (r"\b(search|find|retrieve|documents?|knowledge)\b", "vector-search"),
        (r"\b(history|session|remember|discussed|conversation)\b", "memory-read"),
    ]), min_confidence=0.9)
    router.add_tier(SemanticRouter(min_confidence=0.1), min_confidence=0.3)

    context = RoutingContext(available_skills=registry.list_skills())
    judge = RoutingJudge()

    correct = 0
    total = len(EVAL_DATASET)
    results = []

    for case in EVAL_DATASET:
        decision = router.route(case["query"], context)
        verdict = judge.evaluate(JudgeInput(
            query=case["query"], response=decision.skill_name,
            expectations={"expected_skill": case["expected_skill"]},
        ))
        results.append({
            "query": case["query"], "expected": case["expected_skill"],
            "actual": decision.skill_name, "confidence": decision.confidence,
            "passed": verdict.passed, "rationale": decision.rationale,
        })
        if verdict.passed:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nRouting Accuracy: {correct}/{total} ({accuracy:.0%})")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] '{r['query'][:50]}' -> {r['actual']} (expected {r['expected']}, conf={r['confidence']:.2f})")

    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


if __name__ == "__main__":
    run_eval()
