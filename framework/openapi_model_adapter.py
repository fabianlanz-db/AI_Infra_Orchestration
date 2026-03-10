"""
Generic OpenAPI/HTTP adapter for external model backends.

This adapter implements `ExternalModelClient` and supports most model APIs by:
- posting JSON payloads to an inference endpoint
- extracting response text from a configurable JSON path
- optional health-check endpoint
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from framework.fm_agent_utils import ExternalModelClient, ExternalModelRequest, FmResponse


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

    def __init__(self, config: OpenApiModelConfig) -> None:
        self.config = config

    def build_request_body(self, request: ExternalModelRequest) -> dict[str, Any]:
        """
        Default request schema (OpenAI-style chat payload).
        """
        return {
            "model": self.config.default_model,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }

    def _http_json(self, url: str, method: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body or {}).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", **self.config.headers}
        request = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            content = response.read().decode("utf-8")
            return json.loads(content) if content else {}

    def generate(self, request: ExternalModelRequest) -> FmResponse:
        start = time.perf_counter()
        payload = self.build_request_body(request)
        response_json = self._http_json(
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
        url = self.config.health_url or self.config.inference_url
        method = self.config.health_method if self.config.health_url else self.config.method
        try:
            _ = self._http_json(url=url, method=method, body={} if method.upper() != "GET" else None)
            return True, "External OpenAPI model endpoint reachable"
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            return False, f"External OpenAPI model error: {exc}"
