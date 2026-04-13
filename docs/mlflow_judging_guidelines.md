# MLflow Judging Guidelines (Demo Ready)

Use this page to demonstrate how to judge agent responses with MLflow GenAI evaluators.

## Purpose

Show stakeholders that responses are not only generated, but also scored against explicit quality standards.

## Recommended Judging Dimensions

- **Groundedness**: response stays within retrieved context and does not invent facts.
- **Actionability**: response gives concrete next steps, not generic advice.
- **Safety**: response avoids unsafe actions and calls out risk constraints.
- **Escalation Quality**: response identifies when human escalation is required.
- **Format Compliance**: response follows expected structure (for this demo: Summary, Recommended Actions, Risk Notes).

## Example Guideline Texts (for `Guidelines` scorer)

### 1) Grounded Response

`The response must rely on provided retrieved context. If context is insufficient, it must explicitly say so and avoid unsupported claims.`

### 2) Operational Actionability

`The response must include specific operational actions in a prioritized sequence, with clear immediate next steps.`

### 3) Safety and Escalation

`The response must include safety/risk notes and identify escalation criteria when incident severity is high or uncertainty is material.`

### 4) Structured Output

`The response must contain three sections: Summary, Recommended Actions, and Risk Notes.`

## Suggested Pass/Fail Rubric (Demo)

- **Pass**:
  - No hallucinated facts beyond retrieved context
  - At least 2 concrete actions
  - Includes at least 1 risk/safety note
  - Uses required 3-section structure
- **Fail**:
  - Unsupported claims
  - Missing actionable steps
  - Missing risk/escalation guidance
  - Missing required response structure

## Minimal MLflow Example

```python
from mlflow.genai.scorers import Guidelines

scorers = [
    Guidelines(
        name="grounded_response",
        guidelines=(
            "The response must rely on retrieved context and avoid unsupported claims."
        ),
        model="databricks:/databricks-meta-llama-3-3-70b-instruct",
    ),
    Guidelines(
        name="operational_actionability",
        guidelines=(
            "The response must provide concrete, prioritized operational actions."
        ),
        model="databricks:/databricks-meta-llama-3-3-70b-instruct",
    ),
    Guidelines(
        name="safety_and_escalation",
        guidelines=(
            "The response must include safety notes and escalation criteria."
        ),
        model="databricks:/databricks-meta-llama-3-3-70b-instruct",
    ),
]
```

## Custom Judges (framework/judge_hooks.py)

The repository includes a pluggable judge framework for evaluation criteria that go beyond what built-in MLflow scorers cover.

### JudgeClient Protocol

Any class that implements `name` (property) and `evaluate(JudgeInput) -> JudgeVerdict` satisfies the contract:

```python
from framework.judge_hooks import JudgeClient, JudgeInput, JudgeVerdict

class MyCustomJudge:
    @property
    def name(self) -> str:
        return "my_custom_check"

    def evaluate(self, input: JudgeInput) -> JudgeVerdict:
        passed = "critical" not in input.response.lower()
        return JudgeVerdict(
            judge_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            rationale="Response avoids critical terminology." if passed else "Found 'critical' in response.",
        )
```

### Reference Judges

| Judge | What It Checks | Configurable |
|-------|---------------|-------------|
| `FormatComplianceJudge` | Response contains Summary, Recommended Actions, Risk Notes sections | Section list |
| `LatencyThresholdJudge` | End-to-end latency stays below threshold | `threshold_ms` (default 5000) |
| `GroundednessJudge` | Response terms overlap with retrieved context | `min_overlap_ratio` (default 0.15) |

### Using Custom Judges with MLflow

Wrap any judge as an MLflow scorer via `make_mlflow_scorer`, or use `build_judge_suite` to combine custom judges with built-in scorers:

```python
from framework.judge_hooks import (
    FormatComplianceJudge,
    LatencyThresholdJudge,
    GroundednessJudge,
    build_judge_suite,
)

scorers = build_judge_suite(
    custom_judges=[
        FormatComplianceJudge(),
        LatencyThresholdJudge(threshold_ms=3000),
        GroundednessJudge(),
    ],
    fm_endpoint="databricks-meta-llama-3-3-70b-instruct",
)

results = mlflow.genai.evaluate(data=dataset, predict_fn=predict_fn, scorers=scorers)
```

### Assessment Runner

For a full assessment with all judges and a richer dataset:

```bash
python scripts/build_assessment_dataset.py   # generate 7-case dataset
python scripts/run_assessment.py             # run all judges + built-in scorers
```

## Demo Talk Track (30 seconds)

"We use MLflow judges to score each response against business rules: groundedness, actionability, and safety. Custom judges let teams add domain-specific checks — like format compliance and latency gates — alongside MLflow's built-in scorers. This lets us track quality over time and gate regressions before production."
