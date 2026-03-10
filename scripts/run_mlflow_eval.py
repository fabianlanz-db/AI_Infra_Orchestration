"""
Run baseline MLflow GenAI evaluation for the FM-first app pattern.

Usage:
  python scripts/run_mlflow_eval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.genai
from mlflow.genai.scorers import Correctness, Guidelines

from framework.fm_agent_utils import FmAgentClient
from framework.mlflow_tracing_utils import configure_tracing
from framework.vector_search_utils import VectorSearchClient


def build_predict_fn(fm_client: FmAgentClient, vs_client: VectorSearchClient):
    def predict_fn(query: str):
        retrieval = vs_client.retrieve(query, top_k=5)
        context = "\n\n".join([str(row[6]) for row in retrieval.rows[:5]]) if retrieval.rows else "No context."
        response = fm_client.generate(
            system_prompt=(
                "You are an industrial operations assistant. Use only retrieved context and be concise."
            ),
            user_prompt=f"Question: {query}\n\nRetrieved context:\n{context}",
            temperature=0.2,
        )
        return {"response": response.text}

    return predict_fn


def main() -> None:
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/asml-fm-agent-demo")
    configure_tracing(experiment_name=experiment_name, trace_destination=os.environ.get("MLFLOW_TRACING_DESTINATION"))

    dataset_path = Path("scripts/eval_dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "scripts/eval_dataset.json not found. Run scripts/build_eval_dataset.py first."
        )
    eval_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    fm_client = FmAgentClient()
    vs_client = VectorSearchClient()
    predict_fn = build_predict_fn(fm_client, vs_client)

    scorers = [
        Guidelines(
            name="operational_actionability",
            guidelines=(
                "The response must include clear operational actions and avoid unsupported claims."
            ),
            model=f"databricks:/{fm_client.endpoint_name}",
        ),
        Correctness(model=f"databricks:/{fm_client.endpoint_name}"),
    ]

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    print("Evaluation run_id:", results.run_id)
    print("Evaluation metrics:", results.metrics)


if __name__ == "__main__":
    main()
