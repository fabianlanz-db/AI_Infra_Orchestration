from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any

import mlflow


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
