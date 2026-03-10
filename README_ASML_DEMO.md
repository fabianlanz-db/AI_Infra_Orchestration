# ASML FM Agent Framework Demo

This repo now demonstrates an FM-endpoint-first architecture and reusable framework modules customers can apply across applications and external agents.

## Resources Created

- Catalog/Schema: `fl_demos.asml_external_agent_demo`
- Vector endpoint: `asml_external_agent_vs_ep`
- Vector index: `fl_demos.asml_external_agent_demo.asml_kb_index`
- Lakebase Autoscale project: `projects/asml-external-agent-db`

## Reusable Framework Modules

Root reusable modules:
- `framework/fm_agent_utils.py`
- `framework/vector_search_utils.py`
- `framework/lakebase_utils.py`
- `framework/mlflow_tracing_utils.py`

App-local copies (self-contained Databricks App deployment):
- `apps/asml_showcase_app/framework/*.py`

## 1) Databricks Showcase App (FM-first)

Run locally:

```bash
cd /Users/fabian.lanz/AI_Orchestration/apps/asml_showcase_app
uv pip install -r requirements.txt
uv run streamlit run app.py
```

Key env vars (from `app.yaml`):
- `FM_ENDPOINT_NAME`
- `VS_INDEX_NAME`
- `LAKEBASE_ENDPOINT_RESOURCE`
- `LAKEBASE_HOST`
- `LAKEBASE_DB_NAME`
- `MLFLOW_EXPERIMENT_NAME`

## 2) Deploy Databricks App

```bash
export DATABRICKS_CONFIG_PROFILE=azure-demo
export APP_NAME=asml-external-agent-showcase
export WS_SRC=/Workspace/Users/fabian.lanz@databricks.com/asml_showcase_app

databricks workspace import-dir "./apps/asml_showcase_app" "$WS_SRC" --overwrite -p "$DATABRICKS_CONFIG_PROFILE"
databricks apps deploy "$APP_NAME" --source-code-path "$WS_SRC" -p "$DATABRICKS_CONFIG_PROFILE"
databricks apps get "$APP_NAME" -p "$DATABRICKS_CONFIG_PROFILE" -o json
```

## 3) MLflow Evaluation Starter

Build dataset:

```bash
python scripts/build_eval_dataset.py
```

Run evaluation:

```bash
python scripts/run_mlflow_eval.py
```

In app, click `Verify traces` to confirm tracing output.

## 4) Databricks Resource Bootstrap Scripts

Synthetic data generation:

```bash
uv run python scripts/generate_asml_synthetic_data.py
```

Full bootstrap (UC + Vector Search + Lakebase):

```bash
uv run python scripts/bootstrap_asml_resources.py
```

## 5) External Connectivity Guidelines

See:
- `docs/external_connectivity_guidelines.md`

It includes customer patterns for:
- external app/agent -> Vector Search connectivity
- external app/agent -> Lakebase OAuth + connection lifecycle
- retries/timeouts/token refresh/security hardening
- MLflow tracing + evaluation integration

## Live Demo Flow

1. Ask a maintenance question in app chat.
2. Show retrieved context rows from Vector Search.
3. Show persisted session memory from Lakebase.
4. Click `Verify traces` and show `trace_count`.
5. Run `python scripts/run_mlflow_eval.py` and show evaluation metrics.
