---
name: ai-infra-mlflow-tracing-hooks
description: Instruments AI Infra Orchestration flows with MLflow tracing utilities, verifies traces, and propagates trace context headers to external services. Use when adding trace spans, validating trace visibility, or correlating external API logs with MLflow traces.
---

# AI Infra MLflow Tracing Hooks

Use this skill to add and validate tracing for framework-based agent flows.

## Use When

- User asks to add MLflow tracing to an app/agent flow.
- User needs trace correlation across external services.
- User reports missing traces or wants quick verification.

## Primary Utilities

- `framework/mlflow_tracing_utils.py`
  - `configure_tracing(experiment_name, trace_destination)`
  - `traced(name, span_type)`
  - `verify_traces(max_results)`
  - `build_trace_context_headers(trace_id)`
  - `extract_trace_context_headers(headers)`

## Required Workflow

1. Call `configure_tracing(...)` early in app startup.
2. Wrap key operations with `@traced(...)`.
3. Use `verify_traces(...)` for quick runtime confirmation.
4. For external hops:
   - propagate `x-mlflow-trace-id` via `build_trace_context_headers(...)`.
   - parse/forward inbound headers with `extract_trace_context_headers(...)`.

## Notes

- Use Databricks tracking URI (already handled by utility).
- Keep one clear root span/trace per user prompt when possible.
