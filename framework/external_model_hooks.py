"""
Integration hooks for model APIs running outside Databricks.

This module defines:
- OpenAPI/HTTP external model adapter primitives
- a clear external model adapter contract
- an end-to-end orchestrator flow

It combines:
- Databricks Vector Search retrieval
- external model generation
- Lakebase memory persistence
- optional MLflow trace header propagation

This file is the canonical home for external model integration logic.
`framework/openapi_model_adapter.py` is kept as a compatibility wrapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from framework.fm_agent_utils import FmResponse
from framework.lakebase_utils import LakebaseMemoryStore
from framework.mlflow_tracing_utils import build_trace_context_headers
from framework.vector_search_utils import VectorSearchClient, build_external_retrieval_payload


@dataclass
class ExternalModelRequest:
    """Provider-neutral request payload for external model APIs."""

    system_prompt: str
    user_prompt: str
    temperature: float = 0.2


@runtime_checkable
class ExternalModelClient(Protocol):
    """Adapter contract for model APIs running outside Databricks."""

    def generate(self, request: ExternalModelRequest) -> FmResponse:
        """Execute one model generation request."""

    def health(self) -> tuple[bool, str]:
        """Return backend health tuple `(ok, message)`."""


def generate_with_external_client(
    client: ExternalModelClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> FmResponse:
    """External-model hook for provider-neutral generation."""
    request = ExternalModelRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    return client.generate(request)


@dataclass
class OpenApiModelConfig:
    """
    Configuration for an external HTTP/OpenAPI model endpoint.

    `response_text_path` supports dotted paths with list indexes:
    e.g. `choices.0.message.content` or `output.text`.
    """

    inference_url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    method: str = "POST"
    default_model: str = "external-openapi-model"
    response_text_path: str = "choices.0.message.content"
    health_url: str | None = None
    health_method: str = "GET"
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5


def _extract_by_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


class OpenApiModelClient(ExternalModelClient):
    """
    External model client for OpenAPI-compatible HTTP endpoints.

    Override `build_request_body()` if your endpoint expects a custom schema.
    """

    def __init__(self, config: OpenApiModelConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def build_request_body(self, request: ExternalModelRequest) -> dict[str, Any]:
        """Default request schema (OpenAI-style chat payload)."""
        return {
            "model": self.config.default_model,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }

    def _request_json(
        self, url: str, method: str, body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    json=body if method.upper() != "GET" else None,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status in (408, 425, 429, 500, 502, 503, 504) or isinstance(exc, httpx.RequestError)
                if not retryable or attempt == self.config.max_retries:
                    raise
                time.sleep(self.config.retry_backoff_seconds * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def generate(self, request: ExternalModelRequest) -> FmResponse:
        start = time.perf_counter()
        payload = self.build_request_body(request)
        response_json = self._request_json(
            url=self.config.inference_url,
            method=self.config.method,
            body=payload,
        )
        text = str(_extract_by_path(response_json, self.config.response_text_path))
        latency_ms = int((time.perf_counter() - start) * 1000)
        return FmResponse(
            text=text,
            latency_ms=latency_ms,
            model=self.config.default_model,
        )

    def health(self) -> tuple[bool, str]:
        """Liveness probe. Defaults to GET on ``health_url`` (or inference URL)."""
        url = self.config.health_url or self.config.inference_url
        method = self.config.health_method
        try:
            response = self._client.request(
                method=method.upper(),
                url=url,
                headers=self.config.headers,
                timeout=self.config.timeout_seconds,
            )
            # Treat any non-5xx as reachable for a bare liveness probe.
            if response.status_code < 500:
                return True, f"External OpenAPI model endpoint reachable (HTTP {response.status_code})"
            return False, f"External OpenAPI model returned HTTP {response.status_code}"
        except (httpx.RequestError, ValueError, KeyError, IndexError) as exc:
            return False, f"External OpenAPI model error: {exc}"

    def close(self) -> None:
        self._client.close()


@dataclass
class ExternalAgentTurnResult:
    response: FmResponse
    retrieval_payload: dict[str, Any]
    memory_event_ids: dict[str, int]
    trace_headers: dict[str, str]


def run_external_agent_turn(
    query: str,
    session_id: str,
    external_model_client: ExternalModelClient,
    vector_client: VectorSearchClient | None = None,
    memory_store: LakebaseMemoryStore | None = None,
    top_k: int = 5,
    trace_id: str | None = None,
) -> ExternalAgentTurnResult:
    """
    Reference orchestration hook for external model APIs.

    Steps:
    1) Retrieve context from Databricks Vector Search.
    2) Call external model API through adapter contract.
    3) Persist conversation turn in Lakebase.
    4) Return structured payloads for logging/telemetry.
    """
    vector = vector_client or VectorSearchClient()
    memory = memory_store or LakebaseMemoryStore()
    retrieval = vector.retrieve(query, top_k=top_k)
    retrieval_payload = build_external_retrieval_payload(query=query, retrieval=retrieval, top_k=top_k)

    system_prompt = (
        "You are an industrial operations assistant. Use only provided retrieved context. "
        "If context is insufficient, say so clearly. Keep response concise and actionable."
    )
    user_prompt = (
        f"User question:\n{query}\n\nRetrieved context:\n{retrieval_payload['context_block']}\n\n"
        "Respond with 3 sections: Summary, Recommended Actions, Risk Notes."
    )
    model_response = generate_with_external_client(
        client=external_model_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )

    memory_results = memory.write_exchange(
        session_id=session_id,
        user_message=query,
        assistant_message=model_response.text,
        assistant_metadata={
            "model": model_response.model,
            "top_k": top_k,
            "retrieval_latency_ms": retrieval.latency_ms,
            "inference_backend": "external_model_api",
        },
    )

    headers = build_trace_context_headers(trace_id) if trace_id else {}
    return ExternalAgentTurnResult(
        response=model_response,
        retrieval_payload=retrieval_payload,
        memory_event_ids={
            "user": memory_results["user"].event_id,
            "assistant": memory_results["assistant"].event_id,
        },
        trace_headers=headers,
    )
