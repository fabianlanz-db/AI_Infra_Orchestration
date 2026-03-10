import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from databricks.sdk import WorkspaceClient


@dataclass
class FmResponse:
    text: str
    latency_ms: int
    model: str


@dataclass
class ExternalModelRequest:
    """Provider-neutral request payload for external model APIs."""

    system_prompt: str
    user_prompt: str
    temperature: float = 0.2


@runtime_checkable
class ExternalModelClient(Protocol):
    """
    Adapter contract for model APIs running outside Databricks.

    Implement this in your external agent runtime and pass it to
    `generate_with_external_client()`. This keeps retrieval/memory/tracing
    utilities unchanged while swapping model inference backend.
    """

    def generate(self, request: ExternalModelRequest) -> FmResponse:
        """Execute one model generation request."""

    def health(self) -> tuple[bool, str]:
        """Return backend health tuple `(ok, message)`."""


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
        completion = self._client.chat.completions.create(
            model=self.endpoint_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        output_text = completion.choices[0].message.content if completion.choices else ""
        return FmResponse(text=output_text or "", latency_ms=latency_ms, model=self.endpoint_name)

    def health(self) -> tuple[bool, str]:
        try:
            _ = self.generate(
                system_prompt="You are a health check model.",
                user_prompt="Respond with the word healthy.",
                temperature=0.0,
            )
            return True, "FM endpoint reachable"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"FM endpoint error: {exc}"


def generate_with_external_client(
    client: ExternalModelClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> FmResponse:
    """
    External-model hook: call any non-Databricks model API via adapter.

    Usage:
    1) Implement `ExternalModelClient` around your external API client.
    2) Build prompts as usual from Vector Search context.
    3) Call this function instead of `FmAgentClient.generate(...)`.
    """
    request = ExternalModelRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    return client.generate(request)
