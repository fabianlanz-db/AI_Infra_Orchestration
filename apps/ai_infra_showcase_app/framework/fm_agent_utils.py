import os
import time
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient


@dataclass
class FmResponse:
    text: str
    latency_ms: int
    model: str


class FmAgentClient:
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
        except Exception as exc:
            return False, f"FM endpoint error: {exc}"
