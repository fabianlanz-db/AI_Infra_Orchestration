"""
DSPy adapter layer for Databricks infrastructure.

Provides adapter classes so DSPy programs can use Databricks FM endpoints,
Vector Search, Lakebase memory, and the SkillRegistry without modification.

DSPy is an optional dependency. All adapters are guarded by import checks
and raise clear errors if DSPy is not installed.
"""

from __future__ import annotations

from typing import Any

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.skill_registry import SkillClient, SkillInput, SkillRegistry
from framework.vector_search_utils import VectorSearchClient

try:
    import dspy

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


def _require_dspy() -> None:
    if not _DSPY_AVAILABLE:
        raise ImportError(
            "DSPy is required for this adapter. Install it with: pip install dspy"
        )


class DatabricksLM:
    """DSPy language model backed by a Databricks FM serving endpoint.

    Usage::

        from framework.dspy_adapter import DatabricksLM
        lm = DatabricksLM()
        dspy.configure(lm=lm)
    """

    def __init__(self, fm_client: FmAgentClient | None = None) -> None:
        _require_dspy()
        self._fm = fm_client or FmAgentClient()

    def __call__(self, prompt: str, **kwargs: Any) -> list[str]:
        temperature = kwargs.get("temperature", 0.2)
        response = self._fm.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt,
            temperature=temperature,
        )
        return [response.text]

    @property
    def model_name(self) -> str:
        return self._fm.endpoint_name

    def health(self) -> tuple[bool, str]:
        return self._fm.health()


class DatabricksRetriever:
    """DSPy retriever backed by Databricks Vector Search.

    Usage::

        from framework.dspy_adapter import DatabricksRetriever
        rm = DatabricksRetriever()
        dspy.configure(rm=rm)
    """

    def __init__(self, vector_client: VectorSearchClient | None = None) -> None:
        _require_dspy()
        self._vs = vector_client or VectorSearchClient()

    def __call__(self, query: str, k: int = 5, **kwargs: Any) -> list[Any]:
        result = self._vs.retrieve(query, top_k=k)
        passages = []
        for row in result.rows:
            # Column index 6 is 'content' per vector_search_utils.py column order
            content = row[6] if len(row) > 6 else str(row)
            passages.append(dspy.Prediction(long_text=content))
        return passages

    def health(self) -> tuple[bool, str]:
        return self._vs.health()


class SkillAsTool:
    """Wraps a SkillClient as a callable tool for DSPy programs.

    Usage::

        from framework.dspy_adapter import SkillAsTool
        tool = SkillAsTool(my_skill)
        result = tool("search for maintenance logs")
    """

    def __init__(self, skill: SkillClient) -> None:
        _require_dspy()
        self._skill = skill
        self.name = skill.name
        self.description = skill.definition.description

    def __call__(self, query: str, **kwargs: Any) -> str:
        skill_input = SkillInput(query=query, parameters=kwargs)
        result = self._skill.execute(skill_input)
        output = result.output
        return str(output.get("text", output)) if isinstance(output, dict) else str(output)

    def health(self) -> tuple[bool, str]:
        return self._skill.health()


class LakebaseMemoryModule:
    """DSPy module for reading and writing Lakebase session memory.

    Usage::

        from framework.dspy_adapter import LakebaseMemoryModule
        memory = LakebaseMemoryModule()
        history = memory.read("session-123")
        memory.write("session-123", "user", "Hello")
    """

    def __init__(self, memory_store: LakebaseMemoryStore | None = None) -> None:
        _require_dspy()
        self._store = memory_store or LakebaseMemoryStore()

    def read(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Read conversation history."""
        return self._store.read(session_id=session_id, limit=limit)

    def write(self, session_id: str, role: str, content: str) -> int:
        """Write a message and return the event_id."""
        result = self._store.write(session_id=session_id, role=role, content=content)
        return result.event_id

    def health(self) -> tuple[bool, str]:
        return self._store.health()


def build_dspy_skill_tools(registry: SkillRegistry) -> list[SkillAsTool]:
    """Convert all skills in a registry to DSPy-compatible tools."""
    _require_dspy()
    tools = []
    for defn in registry.list_skills():
        skill = registry.get(defn.name)
        if skill:
            tools.append(SkillAsTool(skill))
    return tools
