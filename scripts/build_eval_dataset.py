"""
Build a starter MLflow GenAI evaluation dataset JSON file.

Output format is compatible with mlflow.genai.evaluate() where each row has:
- inputs (required)
- expectations (recommended for correctness-like checks)
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    output_path = Path("scripts/eval_dataset.json")
    dataset = [
        {
            "inputs": {"query": "How should we triage a critical laser instability alert on EUV tools?"},
            "expectations": {
                "expected_response": (
                    "The response should identify immediate safety checks, suggest diagnostic steps, "
                    "and include clear operational next actions."
                )
            },
        },
        {
            "inputs": {"query": "What should operators do next for recurring vacuum alerts?"},
            "expectations": {
                "expected_response": (
                    "The response should include root-cause oriented diagnostics, mitigation steps, "
                    "and a recommendation for follow-up monitoring."
                )
            },
        },
        {
            "inputs": {"query": "Summarize response priorities for cooling warnings in lithography systems."},
            "expectations": {
                "expected_response": (
                    "The response should provide a concise summary, a prioritized action list, and risk notes."
                )
            },
        },
    ]
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
