from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from framework.router import RoutingDecision
    from framework.skill_registry import SkillDefinition, SkillResult

logger = logging.getLogger(__name__)


def configure_tracing(experiment_name: str | None = None, trace_destination: str | None = None) -> None:
    """Configure MLflow tracing for Databricks environments."""
    mlflow.set_tracking_uri("databricks")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    destination = trace_destination or os.environ.get("MLFLOW_TRACING_DESTINATION")
    if destination:
        os.environ["MLFLOW_TRACING_DESTINATION"] = destination


def traced(name: str, span_type: str | None = None) -> Callable:
    """Decorator wrapper around mlflow.trace for reusable instrumentation."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @mlflow.trace(name=name, span_type=span_type)
        def wrapped(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)

        return wrapped

    return decorator


def verify_traces(max_results: int = 10) -> dict[str, Any]:
    """Return a compact trace verification summary."""
    traces = mlflow.search_traces(max_results=max_results)
    count = len(traces) if traces is not None else 0
    return {"trace_count": count, "verified": count > 0}


def build_trace_context_headers(trace_id: str) -> dict[str, str]:
    """
    Utility for propagating trace context into external services.

    External agents/apps can accept this header and log it alongside
    local telemetry, then map records back to MLflow trace IDs.
    """
    return {"x-mlflow-trace-id": trace_id}


def extract_trace_context_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """
    Normalize inbound/outbound tracing headers for external services.

    External APIs can use this helper to read and forward trace correlation IDs.
    """
    if not headers:
        return {}
    trace_id = headers.get("x-mlflow-trace-id", "")
    return {"x-mlflow-trace-id": trace_id} if trace_id else {}


# --- Observability tag schema ------------------------------------------------
#
# Stable tag namespaces for observability dashboards and downstream
# consumers of MLflow traces. Keep keys in sync with
# docs/observability_tags.md.
#
#   agent.*    identity of the caller (in-workspace or external)
#   routing.*  router decision metadata
#   skill.*    skill execution metadata
#   mcp.*      MCP tool invocation metadata (via AI Gateway-governed servers)


@dataclass
class AgentContext:
    """Identity of the agent invoking the framework."""

    id: str
    origin: str  # "internal" | "external"
    framework: str  # "dspy" | "langgraph" | "openapi" | "custom" | ...
    version: str = "0.0.0"
    principal: str | None = None  # user or service principal


_AGENT_HEADER_KEYS = {
    "id": "x-agent-id",
    "origin": "x-agent-origin",
    "framework": "x-agent-framework",
    "version": "x-agent-version",
    "principal": "x-agent-principal",
}


def _stringify_tags(tags: dict[str, Any]) -> dict[str, str]:
    """Coerce tag values to strings; drop None/empty so traces stay clean."""
    out: dict[str, str] = {}
    for k, v in tags.items():
        if v is None or v == "":
            continue
        out[k] = str(v)
    return out


def _apply_tags(tags: dict[str, Any]) -> None:
    """Apply tags to the active trace. Best-effort: never block business logic."""
    cleaned = _stringify_tags(tags)
    if not cleaned:
        return
    try:
        mlflow.update_current_trace(tags=cleaned)
    except Exception as exc:  # noqa: BLE001 — tagging is best-effort instrumentation
        # No active trace, MLflow misconfigured, or backend unreachable.
        # Log at debug so it's diagnosable without spamming logs.
        logger.debug("Failed to apply trace tags: %s", exc)


def set_agent_tags(ctx: AgentContext) -> None:
    """Tag the active trace with agent identity."""
    _apply_tags({
        "agent.id": ctx.id,
        "agent.origin": ctx.origin,
        "agent.framework": ctx.framework,
        "agent.version": ctx.version,
        "agent.principal": ctx.principal,
    })


def set_routing_tags(
    decision: RoutingDecision,
    tier_name: str,
    *,
    rationale_max_chars: int = 200,
) -> None:
    """Tag the active trace with a router decision.

    ``tier_name`` is the router tier that produced the decision
    (e.g. ``"rule"``, ``"lexical"``, ``"embedding"``, ``"llm"``). For
    ``CompositeRouter``, pass the name of the winning tier so downstream
    dashboards can attribute traffic to specific tiers.
    """
    rationale = (decision.rationale or "")[:rationale_max_chars]
    _apply_tags({
        "routing.tier": tier_name,
        "routing.chosen_skill": decision.skill_name,
        "routing.confidence": round(decision.confidence, 4),
        "routing.latency_ms": decision.latency_ms,
        "routing.alternatives": ",".join(decision.alternatives),
        "routing.rationale": rationale,
    })


def set_skill_tags(result: SkillResult, definition: SkillDefinition) -> None:
    """Tag the active trace with skill execution metadata."""
    _apply_tags({
        "skill.name": result.skill_name,
        "skill.source": definition.source,
        "skill.version": definition.version,
        "skill.latency_ms": result.latency_ms,
    })


def set_mcp_tags(
    *,
    server: str,
    server_type: str,
    server_url: str,
    tool: str,
    latency_ms: int,
    auth_mode: str,
) -> None:
    """Tag the active trace with MCP tool invocation metadata.

    ``server_type`` is ``"managed" | "external" | "custom"``.
    ``auth_mode`` is ``"u2m" | "m2m" | "pat" | "byo"`` (BYO = caller-supplied session).
    """
    _apply_tags({
        "mcp.server": server,
        "mcp.server_type": server_type,
        "mcp.server_url": server_url,
        "mcp.tool": tool,
        "mcp.latency_ms": latency_ms,
        "mcp.auth_mode": auth_mode,
    })


def agent_context_from_headers(headers: dict[str, str] | None) -> AgentContext | None:
    """Reconstruct an ``AgentContext`` from inbound HTTP headers.

    Returns ``None`` if the required identity fields (``id``, ``origin``,
    ``framework``) are absent. Header lookup is case-insensitive.
    """
    if not headers:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    id_ = lower.get(_AGENT_HEADER_KEYS["id"], "")
    origin = lower.get(_AGENT_HEADER_KEYS["origin"], "")
    framework = lower.get(_AGENT_HEADER_KEYS["framework"], "")
    if not (id_ and origin and framework):
        return None
    return AgentContext(
        id=id_,
        origin=origin,
        framework=framework,
        version=lower.get(_AGENT_HEADER_KEYS["version"]) or "0.0.0",
        principal=lower.get(_AGENT_HEADER_KEYS["principal"]) or None,
    )


def build_agent_context_headers(ctx: AgentContext) -> dict[str, str]:
    """Serialize an ``AgentContext`` into outbound HTTP headers."""
    headers = {
        _AGENT_HEADER_KEYS["id"]: ctx.id,
        _AGENT_HEADER_KEYS["origin"]: ctx.origin,
        _AGENT_HEADER_KEYS["framework"]: ctx.framework,
        _AGENT_HEADER_KEYS["version"]: ctx.version,
    }
    if ctx.principal:
        headers[_AGENT_HEADER_KEYS["principal"]] = ctx.principal
    return headers


def extract_agent_turn_context(
    headers: dict[str, str] | None,
) -> tuple[str | None, AgentContext | None]:
    """Convenience: extract both trace ID and agent context from inbound headers."""
    if not headers:
        return None, None
    lower = {k.lower(): v for k, v in headers.items()}
    trace_id = lower.get("x-mlflow-trace-id") or None
    return trace_id, agent_context_from_headers(headers)


@contextmanager
def with_agent_context(ctx: AgentContext | None) -> Iterator[None]:
    """Scope helper for agent runtimes (DSPy, LangGraph, etc.).

    Tags the active trace with agent identity on entry. MLflow trace tags are
    set-once at trace level, so there is no unwind on exit — the context
    manager is purely for scoping intent at call sites.

    Passing ``None`` is a no-op so callers can use this even when identity is
    unknown (e.g. anonymous/local dev).
    """
    if ctx is not None:
        set_agent_tags(ctx)
    yield
