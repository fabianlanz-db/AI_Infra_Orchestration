from framework.skill_registry import (
    SkillDefinition,
    SkillInput,
    SkillRegistry,
    SkillResult,
    keyword_score,
)


class _FakeSkill:
    def __init__(self, name: str, description: str, tags: list[str]) -> None:
        self._name = name
        self._desc = description
        self._tags = tags

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(name=self._name, description=self._desc, tags=self._tags)

    def execute(self, input: SkillInput) -> SkillResult:
        return SkillResult(output={"echo": input.query}, latency_ms=0, skill_name=self._name)

    def health(self) -> tuple[bool, str]:
        return True, "ok"


def _registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(_FakeSkill("search", "Retrieve documents via Vector Search.", ["retrieval", "search"]))
    reg.register(_FakeSkill("memory", "Read conversation history from Lakebase.", ["memory", "history"]))
    reg.register(_FakeSkill("generate", "Generate text with an FM endpoint.", ["llm", "generate"]))
    return reg


def test_register_and_get():
    reg = _registry()
    assert reg.get("search") is not None
    assert reg.get("does-not-exist") is None


def test_list_skills_returns_definitions():
    reg = _registry()
    names = {d.name for d in reg.list_skills()}
    assert names == {"search", "memory", "generate"}


def test_discover_ranks_by_overlap():
    reg = _registry()
    top = reg.discover("find relevant documents", top_k=2)
    assert top[0].name == "search"
    assert len(top) <= 2


def test_keyword_score_zero_when_no_overlap():
    defn = SkillDefinition(name="x", description="unrelated description", tags=[])
    assert keyword_score("", defn) == 0.0


def test_unregister_removes_skill():
    reg = _registry()
    reg.unregister("search")
    assert reg.get("search") is None


def test_external_catalog_payload_shape():
    reg = _registry()
    payload = reg.build_external_skill_catalog_payload()
    assert payload["skill_count"] == 3
    assert {s["name"] for s in payload["skills"]} == {"search", "memory", "generate"}
