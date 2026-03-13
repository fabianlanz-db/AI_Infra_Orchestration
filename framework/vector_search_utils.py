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
        self.index_name = index_name or os.environ.get("VS_INDEX_NAME", "")
        if not self.index_name:
            raise ValueError("VS_INDEX_NAME env var or index_name parameter is required")

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


def format_context_block(rows: list[list[Any]], top_k: int = 5) -> str:
    """
    Convert retrieval rows into a model-ready context block.

    External API hook: pass this string directly in your external model request.
    Expected row order: chunk_id, asset_id, document_type, issue_type, severity,
    subsystem, content, tags.
    """
    selected = rows[:top_k]
    lines = []
    for row in selected:
        lines.append(
            (
                f"chunk_id={row[0]}, asset_id={row[1]}, doc_type={row[2]}, issue_type={row[3]}, "
                f"severity={row[4]}, subsystem={row[5]}, tags={row[7]}\n"
                f"content={row[6]}"
            )
        )
    return "\n\n".join(lines)


def build_external_retrieval_payload(
    query: str,
    retrieval: RetrievalResult,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Build a stable payload for non-Databricks model APIs.

    This gives external runtimes both raw rows and a formatted context block.
    """
    return {
        "query": query,
        "top_k": top_k,
        "retrieval_latency_ms": retrieval.latency_ms,
        "rows": retrieval.rows[:top_k],
        "context_block": format_context_block(retrieval.rows, top_k=top_k),
    }
