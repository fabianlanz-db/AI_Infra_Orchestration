import pytest

from framework.external_model_hooks import _extract_by_path


def test_extract_by_path_dotted():
    payload = {"choices": [{"message": {"content": "hello"}}]}
    assert _extract_by_path(payload, "choices.0.message.content") == "hello"


def test_extract_by_path_flat_key():
    assert _extract_by_path({"text": "hi"}, "text") == "hi"


def test_extract_by_path_missing_key_raises():
    with pytest.raises(KeyError):
        _extract_by_path({"a": 1}, "b")


def test_extract_by_path_invalid_index_raises():
    with pytest.raises(IndexError):
        _extract_by_path({"a": [1]}, "a.5")


# --- agent_context propagation (Part A of ACP integration) -------------------

from unittest.mock import patch  # noqa: E402


def test_run_external_agent_turn_sets_agent_tags_when_context_provided():
    from framework.external_model_hooks import (
        ExternalModelRequest,
        run_external_agent_turn,
    )
    from framework.fm_agent_utils import FmResponse
    from framework.mlflow_tracing_utils import AgentContext

    class _FakeExternal:
        def generate(self, request: ExternalModelRequest) -> FmResponse:
            return FmResponse(text="ok", latency_ms=1, model="fake")

        def health(self):
            return True, "ok"

    class _FakeRetrieval:
        latency_ms = 1
        rows = []

    class _FakeVector:
        def retrieve(self, query, top_k):
            return _FakeRetrieval()

    class _FakeMemoryWriteResult:
        event_id = 1

    class _FakeMemory:
        def write_exchange(self, session_id, user_message, assistant_message, assistant_metadata):
            return {"user": _FakeMemoryWriteResult(), "assistant": _FakeMemoryWriteResult()}

    ctx = AgentContext(id="a", origin="external", framework="openapi")

    with patch("framework.external_model_hooks.set_agent_tags") as mock_agent, \
         patch("framework.external_model_hooks.build_external_retrieval_payload",
               return_value={"context_block": "c"}):
        run_external_agent_turn(
            query="q",
            session_id="s",
            external_model_client=_FakeExternal(),
            vector_client=_FakeVector(),
            memory_store=_FakeMemory(),
            agent_context=ctx,
        )
        assert mock_agent.call_count == 1
        assert mock_agent.call_args[0][0] == ctx
