"""Router protocol and reference implementations for intent-to-skill dispatch.

Provides ``RouterClient`` protocol, four reference routers (rule-based,
lexical, embedding, LLM), a composite cascading router, a ``RoutingJudge``
for evaluation, and a ``run_routed_turn`` orchestrator that combines routing
with skill execution.

Every router emits ``routing.*`` trace tags; ``run_routed_turn`` additionally
emits ``skill.*`` tags after execution and ``agent.*`` tags if an
``AgentContext`` is provided. See ``framework.mlflow_tracing_utils`` for the
full tag schema.
"""
from framework.router._composite import CompositeRouter
from framework.router._core import (
    RouterClient,
    RoutingContext,
    RoutingDecision,
)
from framework.router._orchestration import (
    RoutedTurnResult,
    RoutingJudge,
    run_routed_turn,
)
from framework.router._tiers import (
    EmbeddingRouter,
    LexicalRouter,
    LLMRouter,
    RuleBasedRouter,
    SemanticRouter,
)

__all__ = [
    "CompositeRouter",
    "EmbeddingRouter",
    "LexicalRouter",
    "LLMRouter",
    "RoutedTurnResult",
    "RouterClient",
    "RoutingContext",
    "RoutingDecision",
    "RoutingJudge",
    "RuleBasedRouter",
    "SemanticRouter",
    "run_routed_turn",
]
