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
