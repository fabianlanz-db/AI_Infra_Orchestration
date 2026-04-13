"""
Custom judge hooks for evaluating agent responses.

Provides JudgeClient protocol, three reference judges (FormatCompliance,
LatencyThreshold, Groundedness), an MLflow scorer bridge, and a factory
to build mixed scorer suites for mlflow.genai.evaluate().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import mlflow.genai
from mlflow.genai.scorers import Correctness, Guidelines


@dataclass
class JudgeInput:
    """Standardized input payload for all custom judges."""

    query: str
    response: str
    context: str = ""
    latency_ms: int = 0
    expectations: dict[str, str] = field(default_factory=dict)


@dataclass
class JudgeVerdict:
    """Result from a single judge evaluation."""

    judge_name: str
    passed: bool
    score: float  # 0.0 = fail, 1.0 = pass, intermediate for partial credit
    rationale: str


@runtime_checkable
class JudgeClient(Protocol):
    """Adapter contract for custom judges.

    Any class with a ``name`` property and ``evaluate`` method satisfies
    the contract and can be wrapped as an MLflow scorer via ``make_mlflow_scorer``.
    """

    @property
    def name(self) -> str: ...

    def evaluate(self, input: JudgeInput) -> JudgeVerdict: ...


class FormatComplianceJudge:
    """Check response follows the required 3-section structure (Summary, Recommended Actions, Risk Notes)."""

    REQUIRED_SECTIONS = ["Summary", "Recommended Actions", "Risk Notes"]

    @property
    def name(self) -> str:
        return "format_compliance"

    def evaluate(self, input: JudgeInput) -> JudgeVerdict:
        response_lower = input.response.lower()
        found: list[str] = []
        missing: list[str] = []
        for section in self.REQUIRED_SECTIONS:
            if section.lower() in response_lower:
                found.append(section)
            else:
                missing.append(section)

        passed = len(missing) == 0
        score = len(found) / len(self.REQUIRED_SECTIONS)

        if passed:
            rationale = f"All required sections present: {', '.join(self.REQUIRED_SECTIONS)}."
        else:
            rationale = (
                f"Missing sections: {', '.join(missing)}. "
                f"Found: {', '.join(found) or 'none'}."
            )

        return JudgeVerdict(judge_name=self.name, passed=passed, score=score, rationale=rationale)


class LatencyThresholdJudge:
    """Check that end-to-end latency stays below ``threshold_ms`` (default 5000)."""

    def __init__(self, threshold_ms: int = 5000) -> None:
        self._threshold_ms = threshold_ms

    @property
    def name(self) -> str:
        return "latency_threshold"

    def evaluate(self, input: JudgeInput) -> JudgeVerdict:
        latency = input.latency_ms
        passed = latency <= self._threshold_ms

        if latency <= 0:
            score = 1.0
        elif passed:
            score = 1.0
        else:
            # Degrade linearly: at 2x threshold score is 0.0
            score = max(0.0, 1.0 - (latency - self._threshold_ms) / self._threshold_ms)

        if passed:
            rationale = f"Latency {latency}ms is within {self._threshold_ms}ms threshold."
        else:
            rationale = (
                f"Latency {latency}ms exceeds {self._threshold_ms}ms threshold "
                f"by {latency - self._threshold_ms}ms."
            )

        return JudgeVerdict(judge_name=self.name, passed=passed, score=score, rationale=rationale)


_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "from",
    "this", "that", "with", "they", "will", "each", "make", "like",
    "than", "them", "then", "into", "over", "such", "when", "very",
    "some", "just", "also", "more", "other", "would", "about", "should",
    "these", "their", "which", "could", "does", "most", "what", "only",
})


class GroundednessJudge:
    """Check response is grounded in retrieved context via term-overlap heuristic.

    When no context is provided, checks whether the response hedges appropriately.
    For LLM-based groundedness, use MLflow ``Guidelines`` scorer instead.
    """

    HEDGING_PHRASES = [
        "context does not",
        "no relevant context",
        "insufficient context",
        "not enough information",
        "cannot determine",
        "no information available",
    ]

    def __init__(self, min_overlap_ratio: float = 0.15) -> None:
        self._min_overlap_ratio = min_overlap_ratio

    @property
    def name(self) -> str:
        return "groundedness"

    def evaluate(self, input: JudgeInput) -> JudgeVerdict:
        if not input.context.strip():
            return self._evaluate_no_context(input)

        context_terms = set(_extract_terms(input.context))
        response_terms = set(_extract_terms(input.response))

        if not response_terms:
            return JudgeVerdict(
                judge_name=self.name, passed=True, score=1.0, rationale="Empty response.",
            )

        overlap = response_terms & context_terms
        ratio = len(overlap) / len(response_terms)
        passed = ratio >= self._min_overlap_ratio

        return JudgeVerdict(
            judge_name=self.name,
            passed=passed,
            score=min(1.0, ratio / self._min_overlap_ratio) if not passed else 1.0,
            rationale=(
                f"Term overlap ratio {ratio:.2f} "
                f"({'meets' if passed else 'below'} "
                f"{self._min_overlap_ratio} threshold). "
                f"{len(overlap)}/{len(response_terms)} response terms found in context."
            ),
        )

    def _evaluate_no_context(self, input: JudgeInput) -> JudgeVerdict:
        response_lower = input.response.lower()
        hedges = any(phrase in response_lower for phrase in self.HEDGING_PHRASES)
        return JudgeVerdict(
            judge_name=self.name,
            passed=hedges,
            score=1.0 if hedges else 0.0,
            rationale=(
                "No context provided; response appropriately hedges."
                if hedges
                else "No context provided but response makes claims without hedging."
            ),
        )


def _extract_terms(text: str) -> list[str]:
    """Extract meaningful terms (3+ alpha chars, lowered, stop-words removed)."""
    return [t for t in re.findall(r"[a-zA-Z]{3,}", text.lower()) if t not in _STOP_WORDS]


def make_mlflow_scorer(judge: JudgeClient):
    """Wrap a JudgeClient as an MLflow GenAI scorer for ``mlflow.genai.evaluate()``."""

    @mlflow.genai.scorer(name=judge.name)
    def _scorer(
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any] | str,
        expectations: dict[str, Any] | None = None,
    ) -> float:
        out = outputs if isinstance(outputs, dict) else {"response": str(outputs)}
        judge_input = JudgeInput(
            query=inputs.get("query", ""),
            response=out.get("response", ""),
            context=inputs.get("context", ""),
            latency_ms=out.get("latency_ms", 0),
            expectations=expectations or {},
        )
        verdict = judge.evaluate(judge_input)
        return verdict.score

    return _scorer


def build_judge_suite(
    custom_judges: list[JudgeClient] | None = None,
    fm_endpoint: str | None = None,
    include_builtin_correctness: bool = True,
    include_builtin_guidelines: bool = True,
) -> list:
    """Build a combined scorer suite for ``mlflow.genai.evaluate()``.

    Merges custom JudgeClient judges (wrapped as MLflow scorers) with
    optional built-in MLflow scorers. ``fm_endpoint`` is required for
    built-in LLM scorers (Correctness, Guidelines).
    """
    scorers: list = []

    for judge in custom_judges or []:
        scorers.append(make_mlflow_scorer(judge))

    if fm_endpoint and include_builtin_correctness:
        scorers.append(Correctness(model=f"databricks:/{fm_endpoint}"))

    if fm_endpoint and include_builtin_guidelines:
        scorers.append(
            Guidelines(
                name="operational_actionability",
                guidelines=(
                    "The response must include clear operational actions "
                    "and avoid unsupported claims."
                ),
                model=f"databricks:/{fm_endpoint}",
            ),
        )

    return scorers
