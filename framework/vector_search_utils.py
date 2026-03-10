import os
import time
from dataclasses import dataclass
from typing import Any

from databricks.sdk import WorkspaceClient


@dataclass
class RetrievalResult:
    rows: list[list[Any]]
    latency_ms: int


class VectorSearchClient:
    """Reusable Vector Search utility for apps and external agents."""

    def __init__(self, index_name: str | None = None) -> None:
        self.workspace = WorkspaceClient()
        self.index_name = index_name or os.environ.get(
            "VS_INDEX_NAME", "fl_demos.asml_external_agent_demo.asml_kb_index"
        )

    def retrieve(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        response = self.workspace.vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=[
                "chunk_id",
                "asset_id",
                "document_type",
                "issue_type",
                "severity",
                "subsystem",
                "content",
                "tags",
            ],
            query_text=query_text,
            num_results=top_k,
            query_type="HYBRID",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        rows = response.result.data_array if response.result else []
        return RetrievalResult(rows=rows, latency_ms=latency_ms)

    def health(self) -> tuple[bool, str]:
        try:
            _ = self.workspace.vector_search_indexes.get_index(index_name=self.index_name)
            return True, "Vector Search index reachable"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"Vector Search error: {exc}"
