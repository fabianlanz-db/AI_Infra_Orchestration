"""Cascading meta-router that delegates to a sequence of RouterClient tiers.

Internal module. Consumers should import from ``framework.router``.
"""
from __future__ import annotations

import time

from framework.router._core import (
    RouterClient,
    RoutingContext,
    RoutingDecision,
    tagged,
    tier_name_of,
)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


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
        last_router: RouterClient | None = None
        for router, threshold in self._tiers:
            last_decision = router.route(query, context)
            last_router = router
            if last_decision.confidence >= threshold:
                last_decision.latency_ms = _elapsed_ms(start)
                # Re-tag so the trace reflects composite-adjusted latency, not
                # the tier's internal latency. Winning tier's name preserved.
                return tagged(last_decision, tier_name_of(router))
        total_ms = _elapsed_ms(start)
        if last_decision and last_router:
            last_decision.latency_ms = total_ms
            last_decision.rationale = f"No tier met threshold; using last: {last_decision.rationale}"
            return tagged(last_decision, tier_name_of(last_router))
        return RoutingDecision(
            skill_name="", confidence=0.0,
            rationale="No router tiers configured", latency_ms=total_ms,
        )

    def health(self) -> tuple[bool, str]:
        if not self._tiers:
            return False, "CompositeRouter: no tiers configured"
        statuses = [router.health() for router, _ in self._tiers]
        all_ok = all(ok for ok, _ in statuses)
        summary = ", ".join(msg for _, msg in statuses)
        return all_ok, f"CompositeRouter: {len(self._tiers)} tier(s); {summary}"
