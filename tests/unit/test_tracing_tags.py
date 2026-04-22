"""Unit tests for the observability tag schema.

Covers AgentContext, set_*_tags helpers, header serdes, and the context-manager
helper. Router and orchestrator tagging are covered in test_router.py and
test_external_model_hooks.py respectively.
"""
from __future__ import annotations

from unittest.mock import patch

from framework.mlflow_tracing_utils import (
    AgentContext,
    agent_context_from_headers,
    build_agent_context_headers,
    extract_agent_turn_context,
    set_agent_tags,
    set_mcp_tags,
    set_routing_tags,
    set_skill_tags,
    with_agent_context,
)
from framework.router import RoutingDecision
from framework.skill_registry import SkillDefinition, SkillResult


def _captured_tags(apply_mock) -> dict[str, str]:
    """Extract the tags dict passed to mlflow.update_current_trace."""
    assert apply_mock.call_count == 1, f"expected one call, got {apply_mock.call_count}"
    _, kwargs = apply_mock.call_args
    return kwargs["tags"]


# --- AgentContext & header serdes --------------------------------------------

def test_agent_context_from_headers_requires_core_fields():
    assert agent_context_from_headers(None) is None
    assert agent_context_from_headers({}) is None
    # missing framework
    assert agent_context_from_headers({"x-agent-id": "a", "x-agent-origin": "external"}) is None


def test_agent_context_from_headers_is_case_insensitive():
    headers = {
        "X-Agent-Id": "agent-42",
        "X-Agent-Origin": "external",
        "X-Agent-Framework": "langgraph",
        "X-Agent-Version": "1.2.3",
        "X-Agent-Principal": "sp-1",
    }
    ctx = agent_context_from_headers(headers)
    assert ctx is not None
    assert ctx.id == "agent-42"
    assert ctx.origin == "external"
    assert ctx.framework == "langgraph"
    assert ctx.version == "1.2.3"
    assert ctx.principal == "sp-1"


def test_agent_context_headers_roundtrip():
    ctx = AgentContext(id="x", origin="internal", framework="dspy", version="0.1", principal="u@x")
    headers = build_agent_context_headers(ctx)
    restored = agent_context_from_headers(headers)
    assert restored == ctx


def test_build_agent_context_headers_omits_empty_principal():
    ctx = AgentContext(id="x", origin="internal", framework="dspy")
    headers = build_agent_context_headers(ctx)
    assert "x-agent-principal" not in headers


def test_extract_agent_turn_context_parses_both():
    headers = {
        "x-mlflow-trace-id": "trace-abc",
        "x-agent-id": "a",
        "x-agent-origin": "external",
        "x-agent-framework": "custom",
    }
    trace_id, ctx = extract_agent_turn_context(headers)
    assert trace_id == "trace-abc"
    assert ctx is not None and ctx.id == "a"


def test_extract_agent_turn_context_handles_missing():
    assert extract_agent_turn_context(None) == (None, None)
    assert extract_agent_turn_context({})[0] is None


# --- Tag setters (verify the exact keys dashboards will read) ---------------

@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_set_agent_tags_emits_agent_namespace(mock_apply):
    set_agent_tags(AgentContext(id="a", origin="external", framework="dspy", version="2.0", principal="sp"))
    tags = _captured_tags(mock_apply)
    assert tags == {
        "agent.id": "a",
        "agent.origin": "external",
        "agent.framework": "dspy",
        "agent.version": "2.0",
        "agent.principal": "sp",
    }


@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_set_agent_tags_drops_empty_principal(mock_apply):
    set_agent_tags(AgentContext(id="a", origin="internal", framework="custom"))
    tags = _captured_tags(mock_apply)
    assert "agent.principal" not in tags


@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_set_routing_tags_emits_routing_namespace(mock_apply):
    decision = RoutingDecision(
        skill_name="search",
        confidence=0.8765,
        rationale="x" * 300,
        latency_ms=12,
        alternatives=["memory", "generate"],
    )
    set_routing_tags(decision, tier_name="rule")
    tags = _captured_tags(mock_apply)
    assert tags["routing.tier"] == "rule"
    assert tags["routing.chosen_skill"] == "search"
    assert tags["routing.confidence"] == "0.8765"
    assert tags["routing.latency_ms"] == "12"
    assert tags["routing.alternatives"] == "memory,generate"
    # rationale is truncated to 200 chars
    assert len(tags["routing.rationale"]) == 200


@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_set_skill_tags_emits_skill_namespace(mock_apply):
    defn = SkillDefinition(name="search", description="...", version="1.2.3", source="unity_catalog")
    result = SkillResult(output={"ok": True}, latency_ms=42, skill_name="search")
    set_skill_tags(result, defn)
    tags = _captured_tags(mock_apply)
    assert tags == {
        "skill.name": "search",
        "skill.source": "unity_catalog",
        "skill.version": "1.2.3",
        "skill.latency_ms": "42",
    }


@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_set_mcp_tags_emits_mcp_namespace(mock_apply):
    set_mcp_tags(
        server="uc-funcs",
        server_type="managed",
        server_url="https://host/api/2.0/mcp/functions/cat/sch",
        tool="search",
        latency_ms=99,
        auth_mode="u2m",
    )
    tags = _captured_tags(mock_apply)
    assert tags == {
        "mcp.server": "uc-funcs",
        "mcp.server_type": "managed",
        "mcp.server_url": "https://host/api/2.0/mcp/functions/cat/sch",
        "mcp.tool": "search",
        "mcp.latency_ms": "99",
        "mcp.auth_mode": "u2m",
    }


# --- Error-swallowing behavior -----------------------------------------------

@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace", side_effect=RuntimeError("no active trace"))
def test_set_tags_swallows_mlflow_errors(mock_apply):
    # Must not raise — tagging is best-effort instrumentation.
    set_agent_tags(AgentContext(id="a", origin="internal", framework="custom"))


# --- with_agent_context context manager --------------------------------------

@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_with_agent_context_tags_on_entry(mock_apply):
    ctx = AgentContext(id="a", origin="internal", framework="custom")
    with with_agent_context(ctx):
        pass
    assert mock_apply.call_count == 1


@patch("framework.mlflow_tracing_utils.mlflow.update_current_trace")
def test_with_agent_context_none_is_noop(mock_apply):
    with with_agent_context(None):
        pass
    assert mock_apply.call_count == 0
