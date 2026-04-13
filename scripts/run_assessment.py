"""
Run a full assessment of any ExternalModelClient with pluggable custom judges.

This script demonstrates the end-to-end pattern:
1. Load the assessment dataset (richer than the baseline eval dataset).
2. Build a predict function that calls the FM endpoint with retrieved context.
3. Assemble a scorer suite mixing custom judges and built-in MLflow scorers.
4. Run ``mlflow.genai.evaluate()`` and print results.

Usage:
    python scripts/run_assessment.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.genai

from framework.fm_agent_utils import FmAgentClient
from framework.judge_hooks import (
    FormatComplianceJudge,
    GroundednessJudge,
    LatencyThresholdJudge,
    build_judge_suite,
)
from framework.mlflow_tracing_utils import configure_tracing
from framework.vector_search_utils import VectorSearchClient


def build_predict_fn(fm_client: FmAgentClient, vs_client: VectorSearchClient):
    """Build a predict function that returns response text and latency.

    Unlike the baseline eval predict function, this version also returns
    ``latency_ms`` so the LatencyThresholdJudge can score it.
    """

    def predict_fn(query: str, context: str = ""):
        if not context:
            retrieval = vs_client.retrieve(query, top_k=5)
            context = (
                "\n\n".join([str(row[6]) for row in retrieval.rows[:5]])
                if retrieval.rows
                else "No context."
            )

        response = fm_client.generate(
            system_prompt=(
                "You are an industrial operations assistant. "
                "Use only retrieved context and be concise. "
                "Structure your answer in three sections: Summary, Recommended Actions, Risk Notes."
            ),
            user_prompt=f"Question: {query}\n\nRetrieved context:\n{context}",
            temperature=0.2,
        )
        return {"response": response.text, "latency_ms": response.latency_ms}

    return predict_fn


def main() -> None:
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/ai-infra-fm-agent-demo")
    configure_tracing(
        experiment_name=experiment_name,
        trace_destination=os.environ.get("MLFLOW_TRACING_DESTINATION"),
    )

    dataset_path = Path("scripts/assessment_dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "scripts/assessment_dataset.json not found. "
            "Run scripts/build_assessment_dataset.py first."
        )
    eval_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    fm_client = FmAgentClient()
    vs_client = VectorSearchClient()
    predict_fn = build_predict_fn(fm_client, vs_client)

    custom_judges = [
        FormatComplianceJudge(),
        LatencyThresholdJudge(threshold_ms=5000),
        GroundednessJudge(min_overlap_ratio=0.15),
    ]

    scorers = build_judge_suite(
        custom_judges=custom_judges,
        fm_endpoint=fm_client.endpoint_name,
        include_builtin_correctness=True,
        include_builtin_guidelines=True,
    )

    print(f"Running assessment with {len(scorers)} scorers on {len(eval_dataset)} cases...")
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    print(f"\nAssessment run_id: {results.run_id}")
    print("Assessment metrics:")
    for name, value in sorted(results.metrics.items()):
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
