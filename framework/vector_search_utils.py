from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from databricks.sdk import WorkspaceClient


# Column order returned by retrieve(). Keep in sync with the `columns=` list below.
RETRIEVAL_COLUMNS = (
    "chunk_id",
    "asset_id",
    "document_type",
    "issue_type",
    "severity",
    "subsystem",
    "content",
    "tags",
)


@dataclass
class RetrievalRow:
    """Typed view of a single row returned by ``VectorSearchClient.retrieve``."""

    chunk_id: str
    asset_id: str
    document_type: str
    issue_type: str
    severity: str
    subsystem: str
    content: str
    tags: str

    @classmethod
    def from_raw(cls, row: list[Any]) -> "RetrievalRow":
        # Pad missing trailing columns with empty strings so partial rows don't explode.
        padded = list(row) + [""] * (len(RETRIEVAL_COLUMNS) - len(row))
        return cls(*[str(v) if v is not None else "" for v in padded[: len(RETRIEVAL_COLUMNS)]])

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RetrievalResult:
    rows: list[RetrievalRow]
    latency_ms: int

    @property
    def raw_rows(self) -> list[list[Any]]:
        """Legacy positional view (kept for backwards compatibility)."""
        return [[getattr(r, col) for col in RETRIEVAL_COLUMNS] for r in self.rows]


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
            columns=list(RETRIEVAL_COLUMNS),
            query_text=query_text,
            num_results=top_k,
            query_type="HYBRID",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        raw = response.result.data_array if response.result else []
        rows = [RetrievalRow.from_raw(r) for r in raw]
        return RetrievalResult(rows=rows, latency_ms=latency_ms)

    def health(self) -> tuple[bool, str]:
        try:
            _ = self.workspace.vector_search_indexes.get_index(index_name=self.index_name)
            return True, "Vector Search index reachable"
        except Exception as exc:  # pragma: no cover - runtime integration
            return False, f"Vector Search error: {exc}"


def format_context_block(rows: list[RetrievalRow] | list[list[Any]], top_k: int = 5) -> str:
    """Convert retrieval rows into a model-ready context block.

    Accepts either ``RetrievalRow`` instances (preferred) or legacy positional
    rows for backwards compatibility.
    """
    typed_rows: list[RetrievalRow] = [
        r if isinstance(r, RetrievalRow) else RetrievalRow.from_raw(r)
        for r in rows[:top_k]
    ]
    return "\n\n".join(
        (
            f"chunk_id={r.chunk_id}, asset_id={r.asset_id}, doc_type={r.document_type}, "
            f"issue_type={r.issue_type}, severity={r.severity}, subsystem={r.subsystem}, "
            f"tags={r.tags}\ncontent={r.content}"
        )
        for r in typed_rows
    )


def build_external_retrieval_payload(
    query: str,
    retrieval: RetrievalResult,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build a stable payload for non-Databricks model APIs."""
    selected = retrieval.rows[:top_k]
    return {
        "query": query,
        "top_k": top_k,
        "retrieval_latency_ms": retrieval.latency_ms,
        "rows": [r.as_dict() for r in selected],
        "context_block": format_context_block(selected, top_k=top_k),
    }
