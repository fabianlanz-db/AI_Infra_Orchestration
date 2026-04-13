"""
Build a richer assessment dataset for custom judge evaluation.

Produces ``scripts/assessment_dataset.json`` with 7 cases covering:
- standard operational queries
- safety/escalation scenarios
- edge cases (vague queries, out-of-scope requests)

Output format is compatible with ``mlflow.genai.evaluate()`` and includes
an ``inputs.context`` field so judges can evaluate groundedness.

Usage:
    python scripts/build_assessment_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ASSESSMENT_CASES: list[dict[str, Any]] = [
    # --- Standard operational queries ---
    {
        "inputs": {
            "query": "How should we triage a critical laser instability alert on EUV tools?",
            "context": (
                "EUV laser instability alerts require immediate power-down of the affected "
                "tool, followed by gas-pressure diagnostics and mirror alignment checks. "
                "Escalate to the laser engineering team if instability persists beyond 30 minutes."
            ),
        },
        "expectations": {
            "expected_response": (
                "Identify immediate safety checks, suggest diagnostic steps, "
                "and include clear operational next actions."
            ),
        },
    },
    {
        "inputs": {
            "query": "What should operators do next for recurring vacuum alerts?",
            "context": (
                "Recurring vacuum alerts on lithography chambers indicate possible seal "
                "degradation or pump malfunction. Check chamber door seals, verify turbo "
                "pump RPM, and review leak-rate trend logs for the past 72 hours."
            ),
        },
        "expectations": {
            "expected_response": (
                "Include root-cause oriented diagnostics, mitigation steps, "
                "and a recommendation for follow-up monitoring."
            ),
        },
    },
    {
        "inputs": {
            "query": "Summarize response priorities for cooling warnings in lithography systems.",
            "context": (
                "Cooling warnings in lithography systems are classified by zone: "
                "Zone A (optics) requires immediate pause, Zone B (stage) allows continued "
                "operation with reduced throughput, Zone C (environment) is informational. "
                "All zones require coolant flow verification within 15 minutes."
            ),
        },
        "expectations": {
            "expected_response": (
                "Provide a concise summary, a prioritized action list by zone, "
                "and risk notes for delayed response."
            ),
        },
    },
    # --- Safety / escalation scenarios ---
    {
        "inputs": {
            "query": "A gas leak sensor triggered in the EUV source chamber. What now?",
            "context": (
                "Gas leak events in EUV source chambers are safety-critical. "
                "Immediately evacuate personnel from the tool bay, activate emergency "
                "ventilation, and notify the facility safety officer. Do not attempt "
                "manual intervention until gas concentrations are verified below PEL."
            ),
        },
        "expectations": {
            "expected_response": (
                "Must prioritize personnel safety, include evacuation steps, "
                "and clearly flag escalation to the safety officer."
            ),
        },
    },
    {
        "inputs": {
            "query": "Power supply unit is showing intermittent faults across three tools.",
            "context": (
                "Intermittent PSU faults across multiple tools suggest a facility-level "
                "power quality issue. Check UPS event logs, verify input voltage stability "
                "at the PDU, and coordinate with facilities engineering for grid-side diagnostics."
            ),
        },
        "expectations": {
            "expected_response": (
                "Identify cross-tool correlation, suggest facility-level root-cause "
                "investigation, and recommend coordination with facilities engineering."
            ),
        },
    },
    # --- Edge cases ---
    {
        "inputs": {
            "query": "Tell me about the weather tomorrow.",
            "context": "",
        },
        "expectations": {
            "expected_response": (
                "The model should indicate that this is outside its domain "
                "and that context is insufficient to answer."
            ),
        },
    },
    {
        "inputs": {
            "query": "Give me a general overview of semiconductor manufacturing.",
            "context": (
                "The knowledge base covers operational alerts, maintenance procedures, "
                "and troubleshooting for EUV lithography tools in Facility 3."
            ),
        },
        "expectations": {
            "expected_response": (
                "Acknowledge scope limitation: the system covers operational alerts "
                "and maintenance, not general semiconductor education."
            ),
        },
    },
]


def main() -> None:
    output_path = Path("scripts/assessment_dataset.json")
    output_path.write_text(json.dumps(_ASSESSMENT_CASES, indent=2), encoding="utf-8")
    print(f"Wrote {len(_ASSESSMENT_CASES)} assessment records to {output_path}")


if __name__ == "__main__":
    main()
