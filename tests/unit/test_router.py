from framework.router import (
    CompositeRouter,
    EmbeddingRouter,
    LexicalRouter,
    LLMRouter,
    RoutingContext,
    RoutingDecision,
    RuleBasedRouter,
    SemanticRouter,
)
from framework.skill_registry import SkillDefinition


def _skills() -> list[SkillDefinition]:
    return [
        SkillDefinition(name="search", description="Retrieve documents", tags=["retrieval"]),
        SkillDefinition(name="memory", description="Read history", tags=["memory", "history"]),
        SkillDefinition(name="generate", description="Generate text", tags=["llm"]),
    ]


def _ctx() -> RoutingContext:
    return RoutingContext(available_skills=_skills())


def test_rule_based_router_matches_first_rule():
    r = RuleBasedRouter(rules=[(r"\bfind\b", "search"), (r"\bhistory\b", "memory")])
    decision = r.route("find the docs", _ctx())
    assert decision.skill_name == "search"
    assert decision.confidence == 1.0


def test_rule_based_router_accepts_per_rule_confidence():
    r = RuleBasedRouter(rules=[(r"\bfind\b", "search", 0.3)])
    decision = r.route("find the docs", _ctx())
    assert decision.confidence == 0.3


def test_rule_based_router_falls_back_to_default():
    r = RuleBasedRouter(rules=[(r"\bfind\b", "search")], default_skill="generate")
    decision = r.route("something random", _ctx())
    assert decision.skill_name == "generate"
    assert decision.confidence == 0.5


def test_lexical_router_picks_best_overlap():
    r = LexicalRouter(min_confidence=0.1)
    decision = r.route("retrieve documents from the index", _ctx())
    assert decision.skill_name == "search"


def test_lexical_router_returns_empty_when_no_overlap():
    r = LexicalRouter()
    decision = r.route("asdf qwerty", _ctx())
    assert decision.skill_name == ""
    assert decision.confidence == 0.0


def test_semantic_router_alias_preserved():
    assert SemanticRouter is LexicalRouter


class _FakeEmbedder:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            lt = t.lower()
            out.append([
                1.0 if "retrieve" in lt or "search" in lt else 0.0,
                1.0 if "history" in lt or "memory" in lt else 0.0,
                1.0 if "generate" in lt or "llm" in lt else 0.0,
            ])
        return out


def test_embedding_router_picks_best_cosine():
    r = EmbeddingRouter(embedder=_FakeEmbedder(), min_confidence=0.5)
    decision = r.route("retrieve me some stuff", _ctx())
    assert decision.skill_name == "search"
    assert 0.0 < decision.confidence <= 1.0


def test_embedding_router_below_threshold_returns_empty():
    # Query has no terms that map to any non-zero embedding dim, so cosine=0.
    r = EmbeddingRouter(embedder=_FakeEmbedder(), min_confidence=0.5)
    decision = r.route("totally unrelated request", _ctx())
    assert decision.skill_name == ""
    assert decision.confidence == 0.0


def test_composite_router_cascades_and_short_circuits():
    rule = RuleBasedRouter(rules=[(r"\bfind\b", "search")])
    lex = LexicalRouter(min_confidence=0.1)
    comp = CompositeRouter()
    comp.add_tier(rule, min_confidence=0.9)
    comp.add_tier(lex, min_confidence=0.3)

    # Rule tier hits first.
    decision = comp.route("find docs", _ctx())
    assert decision.skill_name == "search"

    # Rule tier returns default below threshold → lexical runs.
    decision = comp.route("read history", _ctx())
    assert decision.skill_name == "memory"


def test_composite_router_health_aggregates():
    rule = RuleBasedRouter(rules=[])
    lex = LexicalRouter()
    comp = CompositeRouter()
    comp.add_tier(rule, min_confidence=0.9)
    comp.add_tier(lex, min_confidence=0.3)
    ok, msg = comp.health()
    assert ok is True
    assert "2 tier(s)" in msg


def test_composite_router_health_with_no_tiers():
    comp = CompositeRouter()
    ok, msg = comp.health()
    assert ok is False
    assert "no tiers" in msg


class _FakeFm:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        from types import SimpleNamespace

        return SimpleNamespace(text=self._text, latency_ms=1, model="fake")

    def health(self):
        return True, "fake"


def test_llm_router_parses_valid_json():
    fm = _FakeFm('{"skill_name":"search","confidence":0.92,"rationale":"r"}')
    decision = LLMRouter(fm).route("q", _ctx())
    assert decision.skill_name == "search"
    assert decision.confidence == 0.92


def test_llm_router_unknown_skill_returns_empty():
    fm = _FakeFm('{"skill_name":"does-not-exist","confidence":0.9,"rationale":"r"}')
    decision = LLMRouter(fm).route("q", _ctx())
    assert decision.skill_name == ""
    assert decision.confidence == 0.0


def test_llm_router_parse_failure_returns_empty():
    fm = _FakeFm("not json at all")
    decision = LLMRouter(fm).route("q", _ctx())
    assert decision.skill_name == ""
    assert decision.confidence == 0.0


# --- Observability tag emission (Part A of ACP integration) ------------------
# These tests verify that routers emit routing.* tags with the correct tier_name
# so ACP / dashboards can pivot traffic by tier.

from unittest.mock import patch  # noqa: E402


def _captured_tier(mock) -> str:
    """Return the tier_name passed to the last set_routing_tags call."""
    assert mock.call_count >= 1
    args, kwargs = mock.call_args
    # set_routing_tags(decision, tier_name, *, rationale_max_chars=...)
    return args[1] if len(args) >= 2 else kwargs.get("tier_name", "")


@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_rule_based_router_tags_with_rule_tier(mock_tags):
    RuleBasedRouter(rules=[(r"\bfind\b", "search")]).route("find it", _ctx())
    assert _captured_tier(mock_tags) == "rule"


@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_lexical_router_tags_with_lexical_tier(mock_tags):
    LexicalRouter(min_confidence=0.1).route("retrieve documents", _ctx())
    assert _captured_tier(mock_tags) == "lexical"


@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_embedding_router_tags_with_embedding_tier(mock_tags):
    EmbeddingRouter(embedder=_FakeEmbedder(), min_confidence=0.5).route("retrieve stuff", _ctx())
    assert _captured_tier(mock_tags) == "embedding"


@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_llm_router_tags_with_llm_tier(mock_tags):
    fm = _FakeFm('{"skill_name":"search","confidence":0.9,"rationale":"r"}')
    LLMRouter(fm).route("q", _ctx())
    assert _captured_tier(mock_tags) == "llm"


@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_custom_tier_name_flows_into_tags(mock_tags):
    # Callers can override the default tier_name — useful for A/B testing two
    # variants of the same router class.
    RuleBasedRouter(rules=[(r"\bfind\b", "search")], tier_name="rule-v2").route("find it", _ctx())
    assert _captured_tier(mock_tags) == "rule-v2"


@patch("framework.router._composite.tagged", side_effect=lambda d, _: d)
@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_composite_router_retags_with_winning_tier(mock_tier_tags, mock_comp_tags):
    rule = RuleBasedRouter(rules=[(r"\bfind\b", "search")])
    lex = LexicalRouter(min_confidence=0.1)
    comp = CompositeRouter()
    comp.add_tier(rule, min_confidence=0.9)
    comp.add_tier(lex, min_confidence=0.3)

    # Rule tier wins → inner tier tags first (via _tiers.tagged), then composite
    # re-tags with adjusted latency (via _composite.tagged). Both call sites
    # receive tier_name="rule".
    comp.route("find docs", _ctx())
    assert mock_tier_tags.call_args[0][1] == "rule"
    assert mock_comp_tags.call_args[0][1] == "rule"


@patch("framework.router._composite.tagged", side_effect=lambda d, _: d)
@patch("framework.router._tiers.tagged", side_effect=lambda d, _: d)
def test_composite_router_retags_with_last_tier_on_fallback(mock_tier_tags, mock_comp_tags):
    # No tier meets its threshold → composite falls back to last tier's decision
    # and re-tags with that tier's name.
    rule = RuleBasedRouter(rules=[], default_skill="generate")  # returns confidence=0.5
    lex = LexicalRouter(min_confidence=0.99)  # will also fail high threshold
    comp = CompositeRouter()
    comp.add_tier(rule, min_confidence=0.9)
    comp.add_tier(lex, min_confidence=0.9)

    comp.route("asdf qwerty", _ctx())
    # Composite re-tag should use the LAST tier's name ("lexical").
    assert mock_comp_tags.call_args[0][1] == "lexical"


# --- run_routed_turn skill tagging -------------------------------------------

def test_run_routed_turn_tags_skill_after_execution():
    from framework.router import run_routed_turn
    from framework.skill_registry import SkillDefinition, SkillInput, SkillResult, SkillRegistry

    class _FakeSkill:
        @property
        def name(self) -> str:
            return "generate"

        @property
        def definition(self) -> SkillDefinition:
            return SkillDefinition(name="generate", description="d", version="1.0", source="local")

        def execute(self, _: SkillInput) -> SkillResult:
            return SkillResult(output={"text": "ok"}, latency_ms=5, skill_name="generate")

        def health(self):
            return True, "ok"

    registry = SkillRegistry()
    registry.register(_FakeSkill())
    rule = RuleBasedRouter(rules=[(r".*", "generate")])

    with patch("framework.router._orchestration.set_skill_tags") as mock_tags:
        run_routed_turn("q", session_id="s1", router=rule, registry=registry)
        assert mock_tags.call_count == 1
        args, _ = mock_tags.call_args
        result, definition = args
        assert result.skill_name == "generate"
        assert definition.source == "local"


def test_run_routed_turn_sets_agent_tags_when_context_provided():
    from framework.mlflow_tracing_utils import AgentContext
    from framework.router import run_routed_turn
    from framework.skill_registry import SkillDefinition, SkillInput, SkillResult, SkillRegistry

    class _FakeSkill:
        @property
        def name(self):
            return "generate"

        @property
        def definition(self):
            return SkillDefinition(name="generate", description="d", source="local")

        def execute(self, _):
            return SkillResult(output={}, latency_ms=1, skill_name="generate")

        def health(self):
            return True, "ok"

    registry = SkillRegistry()
    registry.register(_FakeSkill())
    rule = RuleBasedRouter(rules=[(r".*", "generate")])
    ctx = AgentContext(id="a", origin="external", framework="langgraph")

    with patch("framework.router._orchestration.set_agent_tags") as mock_agent:
        run_routed_turn("q", session_id="s", router=rule, registry=registry, agent_context=ctx)
        assert mock_agent.call_count == 1
        assert mock_agent.call_args[0][0] == ctx
