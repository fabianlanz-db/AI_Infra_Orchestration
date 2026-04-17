from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Retry policy for transient FM endpoint failures (429 / 5xx / timeout).
_MAX_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 0.5
_RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
}


@dataclass
class FmResponse:
    text: str
    latency_ms: int
    model: str


class FmAgentClient:
    """Databricks FM endpoint chat client using WorkspaceClient auth."""

    def __init__(self, endpoint_name: str | None = None) -> None:
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name or os.environ.get(
            "FM_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct"
        )
        self._client = self.workspace.serving_endpoints.get_open_ai_client(timeout=60.0)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> FmResponse:
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                completion = self._client.chat.completions.create(
                    model=self.endpoint_name,
                    temperature=temperature,
                    messages=messages,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                output_text = completion.choices[0].message.content if completion.choices else ""
                return FmResponse(text=output_text or "", latency_ms=latency_ms, model=self.endpoint_name)
            except Exception as exc:
                last_exc = exc
                if type(exc).__name__ not in _RETRYABLE_EXCEPTION_NAMES or attempt == _MAX_ATTEMPTS - 1:
                    raise
                backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "FM endpoint call failed (%s). Retrying in %.2fs (attempt %d/%d)",
                    type(exc).__name__, backoff, attempt + 1, _MAX_ATTEMPTS,
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def health(self) -> tuple[bool, str]:
        """Liveness check via endpoint metadata (no inference call)."""
        try:
            endpoint = self.workspace.serving_endpoints.get(name=self.endpoint_name)
            state = getattr(endpoint.state, "ready", None) if endpoint.state else None
            ready = str(state).upper() == "READY" if state is not None else True
            return ready, f"FM endpoint {self.endpoint_name}: {state or 'reachable'}"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"FM endpoint error: {exc}"

    def deep_health(self) -> tuple[bool, str]:
        """End-to-end health check via a minimal completion. Costs tokens."""
        try:
            _ = self.generate(
                system_prompt="You are a health check model.",
                user_prompt="Respond with the word healthy.",
                temperature=0.0,
            )
            return True, "FM endpoint reachable (deep probe)"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"FM endpoint error: {exc}"
