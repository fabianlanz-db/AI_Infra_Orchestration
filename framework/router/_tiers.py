"""Reference router implementations and the cascading CompositeRouter.

Internal module. Consumers should import from ``framework.router``.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from framework.router._core import (
    RoutingContext,
    RoutingDecision,
    tagged,
)
from framework.skill_registry import SkillDefinition, keyword_score


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RuleBasedRouter:
    """Keyword/pattern router. Rules map regex patterns to ``(skill, confidence)``.

    Rules may be passed as 2-tuples ``(pattern, skill)`` — confidence defaults
    to ``default_rule_confidence`` — or 3-tuples ``(pattern, skill, confidence)``
    for fine-grained control so ambiguous rules don't block downstream tiers.
    First match wins.
    """

    def __init__(
        self,
        rules: list[tuple] | None = None,
        default_skill: str = "generate",
        default_rule_confidence: float = 1.0,
        tier_name: str = "rule",
    ) -> None:
        self._rules: list[tuple[re.Pattern[str], str, float]] = []
        for rule in rules or []:
            if len(rule) == 2:
                pattern, skill = rule
                confidence = default_rule_confidence
            elif len(rule) == 3:
                pattern, skill, confidence = rule
            else:
                raise ValueError(f"Rule must be 2- or 3-tuple, got {rule!r}")
            self._rules.append((re.compile(pattern, re.IGNORECASE), skill, float(confidence)))
        self._default = default_skill
        self.tier_name = tier_name

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        for pattern, skill_name, confidence in self._rules:
            if pattern.search(query):
                return tagged(
                    RoutingDecision(
                        skill_name=skill_name,
                        confidence=confidence,
                        rationale=f"Matched rule pattern: {pattern.pattern}",
                        latency_ms=_elapsed_ms(start),
                    ),
                    self.tier_name,
                )
        return tagged(
            RoutingDecision(
                skill_name=self._default,
                confidence=0.5,
                rationale=f"No rule matched; defaulting to {self._default}",
                latency_ms=_elapsed_ms(start),
            ),
            self.tier_name,
        )

    def health(self) -> tuple[bool, str]:
        return True, f"RuleBasedRouter: {len(self._rules)} rule(s)"


class LexicalRouter:
    """Skill-description similarity router using keyword overlap scoring.

    This is lexical (bag-of-words term overlap), not semantic — use
    ``EmbeddingRouter`` for true semantic matching via an embedding model.
    """

    def __init__(self, min_confidence: float = 0.1, tier_name: str = "lexical") -> None:
        self._min_confidence = min_confidence
        self.tier_name = tier_name

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        skills = context.available_skills
        if not skills:
            return tagged(
                RoutingDecision(
                    skill_name="", confidence=0.0, rationale="No skills available",
                    latency_ms=_elapsed_ms(start),
                ),
                self.tier_name,
            )
        scored = sorted(
            [(s, keyword_score(query, s)) for s in skills],
            key=lambda p: p[1], reverse=True,
        )
        best_skill, best_score = scored[0]
        if best_score == 0.0:
            return tagged(
                RoutingDecision(
                    skill_name="", confidence=0.0,
                    rationale="No keyword overlap with any skill",
                    latency_ms=_elapsed_ms(start),
                ),
                self.tier_name,
            )
        alternatives = [s.name for s, sc in scored[1:3] if sc > 0]
        conf = min(1.0, best_score)
        below = best_score < self._min_confidence
        rationale = (
            f"Below threshold ({best_score:.2f} < {self._min_confidence})" if below
            else f"Lexical match {best_score:.2f} for '{best_skill.name}'"
        )
        return tagged(
            RoutingDecision(
                skill_name=best_skill.name, confidence=conf, rationale=rationale,
                alternatives=alternatives, latency_ms=_elapsed_ms(start),
            ),
            self.tier_name,
        )

    def health(self) -> tuple[bool, str]:
        return True, "LexicalRouter active"


# Back-compat alias. Old code and docs referenced ``SemanticRouter``; keep it
# resolving to the lexical implementation until callers migrate.
SemanticRouter = LexicalRouter


class EmbeddingRouter:
    """True semantic router using an embedding model for query/skill similarity.

    Expects ``embedder`` to be a callable ``(texts: list[str]) -> list[list[float]]``.
    Embeds all skill definitions once and caches vectors; re-embeds when the
    available-skills set changes.
    """

    def __init__(
        self,
        embedder: Any,
        min_confidence: float = 0.25,
        tier_name: str = "embedding",
    ) -> None:
        self._embedder = embedder
        self._min_confidence = min_confidence
        self._cache_key: tuple[str, ...] = ()
        self._cached_vectors: list[list[float]] = []
        self.tier_name = tier_name

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

    def _ensure_cached(self, skills: list[SkillDefinition]) -> None:
        key = tuple(f"{s.name}:{s.description}" for s in skills)
        if key == self._cache_key and self._cached_vectors:
            return
        texts = [f"{s.name}. {s.description}. tags: {', '.join(s.tags)}" for s in skills]
        self._cached_vectors = self._embedder(texts)
        self._cache_key = key

    def route(self, query: str, context: RoutingContext) -> RoutingDecision:
        start = time.perf_counter()
        skills = context.available_skills
        if not skills:
            return tagged(
                RoutingDecision(
                    skill_name="", confidence=0.0, rationale="No skills available",
                    latency_ms=_elapsed_ms(start),
                ),
                self.tier_name,
            )
        self._ensure_cached(skills)
        [query_vec] = self._embedder([query])
        scored = sorted(
            [(skills[i], self._cosine(query_vec, v)) for i, v in enumerate(self._cached_vectors)],
            key=lambda p: p[1], reverse=True,
        )
        best_skill, best_score = scored[0]
        passed = best_score >= self._min_confidence
        alternatives = [s.name for s, sc in scored[1:3] if sc > 0]
        return tagged(
            RoutingDecision(
                skill_name=best_skill.name if passed else "",
                confidence=max(0.0, min(1.0, best_score)),
                rationale=(
                    f"Embedding similarity {best_score:.3f} for '{best_skill.name}'"
                    if passed else f"Below threshold ({best_score:.3f} < {self._min_confidence})"
                ),
                alternatives=alternatives,
                latency_ms=_elapsed_ms(start),
            ),
            self.tier_name,
        )

    def health(self) -> tuple[bool, str]:
        return True, "EmbeddingRouter active"


class LLMRouter:
    """LLM-based router using an FM endpoint for intent classification."""

    def __init__(self, fm_client: Any, tier_name: str = "llm") -> None:
        self._fm = fm_client
        self.tier_name = tier_name

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
        return tagged(
            self._parse_response(response.text, skills, _elapsed_ms(start)),
            self.tier_name,
        )

    def _parse_response(
        self, text: str, skills: list[SkillDefinition], latency_ms: int,
    ) -> RoutingDecision:
        try:
            cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
            data = json.loads(cleaned)
            skill_name = data.get("skill_name", "")
            valid_names = {s.name for s in skills}
            if skill_name not in valid_names:
                return RoutingDecision(
                    skill_name="", confidence=0.0,
                    rationale=f"LLM returned unknown skill '{skill_name}'",
                    latency_ms=latency_ms,
                )
            return RoutingDecision(
                skill_name=skill_name,
                confidence=float(data.get("confidence", 0.8)),
                rationale=data.get("rationale", "LLM routing decision"),
                parameters=data.get("parameters", {}),
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return RoutingDecision(
                skill_name="", confidence=0.0,
                rationale="LLM response parsing failed",
                latency_ms=latency_ms,
            )

    def health(self) -> tuple[bool, str]:
        return self._fm.health()


