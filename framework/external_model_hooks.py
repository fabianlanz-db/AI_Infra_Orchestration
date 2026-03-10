"""
Integration hooks for model APIs running outside Databricks.

This module defines a clear adapter contract and an example orchestrator flow
that combines:
- Databricks Vector Search retrieval
- external model generation
- Lakebase memory persistence
- optional MLflow trace header propagation
"""

from dataclasses import dataclass
from typing import Any

from framework.fm_agent_utils import ExternalModelClient, FmResponse, generate_with_external_client
from framework.lakebase_utils import LakebaseMemoryStore
from framework.mlflow_tracing_utils import build_trace_context_headers
from framework.vector_search_utils import VectorSearchClient, build_external_retrieval_payload


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
