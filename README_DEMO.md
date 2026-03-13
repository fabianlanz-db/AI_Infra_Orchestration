# AI Infra Orchestration Demo

This repo demonstrates an FM-endpoint-first architecture and reusable framework modules for connecting any agent runtime to Databricks AI infrastructure.

## Resources Created

- Catalog/Schema: configurable via `DEMO_CATALOG` / `DEMO_SCHEMA` env vars (defaults: `fl_demos.ai_infra_agent_demo`)
- Vector endpoint: `ai_infra_agent_vs_ep`
- Vector index: `fl_demos.ai_infra_agent_demo.ai_infra_kb_index`
- Lakebase Autoscale project: `projects/ai-infra-agent-db`

## Reusable Framework Modules

Root reusable modules:
- `framework/fm_agent_utils.py`
- `framework/vector_search_utils.py`
- `framework/lakebase_utils.py`
- `framework/mlflow_tracing_utils.py`

App-local copies (self-contained Databricks App deployment):
- `apps/ai_infra_showcase_app/framework/*.py`

## 1) Databricks Showcase App (FM-first)

Run locally:

```bash
cd apps/ai_infra_showcase_app
uv pip install -r requirements.txt
uv run streamlit run app.py
```

## 2) Deploy Databricks App

```bash
export DATABRICKS_CONFIG_PROFILE=azure-demo
export APP_NAME=ai-infra-showcase
export WS_SRC=/Workspace/Users/$YOUR_USER/ai_infra_showcase_app

databricks workspace import-dir "./apps/ai_infra_showcase_app" "$WS_SRC" --overwrite -p "$DATABRICKS_CONFIG_PROFILE"
databricks apps deploy "$APP_NAME" --source-code-path "$WS_SRC" -p "$DATABRICKS_CONFIG_PROFILE"
databricks apps get "$APP_NAME" -p "$DATABRICKS_CONFIG_PROFILE" -o json
```

## 3) Evaluation Starter

```bash
python scripts/build_eval_dataset.py
python scripts/run_mlflow_eval.py
```

## 4) Resource Bootstrap

```bash
uv run python scripts/generate_ai_infra_synthetic_data.py
uv run python scripts/bootstrap_ai_infra_resources.py
```

## 5) External Connectivity Guidelines

See `docs/external_connectivity_guidelines.md` for production connectivity, auth, reliability, and observability patterns.
