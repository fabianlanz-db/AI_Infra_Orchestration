import os
from collections.abc import Callable
from functools import wraps
from typing import Any

import mlflow


def configure_tracing(experiment_name: str | None = None, trace_destination: str | None = None) -> None:
    mlflow.set_tracking_uri("databricks")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    destination = trace_destination or os.environ.get("MLFLOW_TRACING_DESTINATION")
    if destination:
        os.environ["MLFLOW_TRACING_DESTINATION"] = destination


def traced(name: str, span_type: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @mlflow.trace(name=name, span_type=span_type)
        def wrapped(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)

        return wrapped

    return decorator


def verify_traces(max_results: int = 10) -> dict[str, Any]:
    traces = mlflow.search_traces(max_results=max_results)
    count = len(traces) if traces is not None else 0
    return {"trace_count": count, "verified": count > 0}


def build_trace_context_headers(trace_id: str) -> dict[str, str]:
    return {"x-mlflow-trace-id": trace_id}
