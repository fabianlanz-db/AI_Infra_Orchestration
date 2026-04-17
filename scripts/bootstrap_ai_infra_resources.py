"""
Bootstrap Databricks resources for the AI Infra Orchestration demo.

Creates/updates:
- UC schema and synthetic Delta tables (configurable via DEMO_CATALOG / DEMO_SCHEMA)
- Vector Search endpoint + Delta Sync index
- Lakebase Autoscaling project

Idempotent: safe to re-run. Pass ``--force-tables`` to recreate the Delta
tables; without it, existing tables are preserved.
"""

import argparse
import logging
import os

from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound, ResourceAlreadyExists
from databricks.sdk.service.postgres import Project, ProjectSpec
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

logger = logging.getLogger(__name__)


CATALOG = os.environ.get("DEMO_CATALOG", "fl_demos")
SCHEMA = os.environ.get("DEMO_SCHEMA", "ai_infra_agent_demo")
VS_ENDPOINT = os.environ.get("DEMO_VS_ENDPOINT", "ai_infra_agent_vs_ep")
VS_INDEX_NAME = os.environ.get("DEMO_VS_INDEX_NAME", "ai_infra_kb_index")
VS_INDEX = f"{CATALOG}.{SCHEMA}.{VS_INDEX_NAME}"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.ai_infra_kb_chunks"
LAKEBASE_PROJECT_ID = os.environ.get("DEMO_LAKEBASE_PROJECT_ID", "ai-infra-agent-db")
LAKEBASE_DISPLAY_NAME = os.environ.get("DEMO_LAKEBASE_DISPLAY_NAME", "External Agent DB")


SCHEMA_DDL = f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA};"


def _tables_ddl(create_verb: str) -> str:
    return f"""
{create_verb} TABLE {CATALOG}.{SCHEMA}.ai_infra_assets USING DELTA AS
SELECT
  concat('ASSET-', lpad(cast(id + 1 as string), 6, '0')) AS asset_id,
  element_at(array('EUVScanner','DUVScanner','MetrologySystem','EtchModule','LithographyTrack'), cast(rand() * 5 as int) + 1) AS asset_type,
  element_at(array('Optics','WaferHandling','VacuumSystem','ThermalControl','LaserSubsystem'), cast(rand() * 5 as int) + 1) AS subsystem,
  element_at(array('Veldhoven','Berlin','SanDiego','Hsinchu','Phoenix'), cast(rand() * 5 as int) + 1) AS facility,
  element_at(array('active','maintenance','degraded'), cast(rand() * 3 as int) + 1) AS asset_state,
  date_sub(current_date(), cast(rand() * 2000 as int)) AS installed_date,
  current_timestamp() AS ingested_at
FROM range(500);

{create_verb} TABLE {CATALOG}.{SCHEMA}.ai_infra_kb_chunks USING DELTA AS
SELECT
  concat('CH-', lpad(cast(id + 1 as string), 8, '0')) AS chunk_id,
  concat('DOC-', lpad(cast(cast(rand() * 900 as int) + 1 as string), 6, '0')) AS document_id,
  concat('ASSET-', lpad(cast(cast(rand() * 500 as int) + 1 as string), 6, '0')) AS asset_id,
  element_at(array('runbook','incident_report','sop','engineering_note','root_cause_analysis'), cast(rand() * 5 as int) + 1) AS document_type,
  element_at(array('alignment_drift','vacuum_alert','laser_instability','cooling_warning','sensor_fault'), cast(rand() * 5 as int) + 1) AS issue_type,
  element_at(array('low','medium','high','critical'), cast(rand() * 4 as int) + 1) AS severity,
  element_at(array('Optics','WaferHandling','VacuumSystem','ThermalControl','LaserSubsystem'), cast(rand() * 5 as int) + 1) AS subsystem,
  concat('Knowledge chunk ', cast(id as string), ' describing diagnostics, probable root cause, stepwise mitigation, safety constraints, and part replacement guidance for industrial operations.') AS content,
  concat_ws(',', 'ai-infra', 'external-agent', 'ops-demo') AS tags,
  timestampadd(HOUR, -cast(rand() * 24 * 365 as int), current_timestamp()) AS source_ts,
  current_timestamp() AS ingested_at
FROM range(8000);
"""


def _table_exists(conn: "sql.Connection", fqn: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(f"SHOW TABLES IN {CATALOG}.{SCHEMA}")
        rows = cur.fetchall()
    name = fqn.split(".")[-1]
    return any(r[1] == name for r in rows)


def execute_statements(conn: sql.Connection, sql_text: str) -> None:
    with conn.cursor() as cur:
        for stmt in [s.strip() for s in sql_text.split(";") if s.strip()]:
            cur.execute(stmt)


def _ensure_vs_endpoint(workspace: WorkspaceClient) -> None:
    try:
        workspace.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
        logger.info("Vector Search endpoint %s already exists", VS_ENDPOINT)
    except NotFound:
        workspace.vector_search_endpoints.create_endpoint(
            name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD,
        )
        logger.info("Created Vector Search endpoint %s", VS_ENDPOINT)


def _ensure_vs_index(workspace: WorkspaceClient) -> None:
    try:
        workspace.vector_search_indexes.get_index(index_name=VS_INDEX)
        logger.info("Vector Search index %s already exists", VS_INDEX)
        return
    except NotFound:
        pass

    vs_spec = DeltaSyncVectorIndexSpecRequest(
        source_table=SOURCE_TABLE,
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name="content",
                embedding_model_endpoint_name="databricks-gte-large-en",
            )
        ],
        pipeline_type=PipelineType.TRIGGERED,
        columns_to_sync=[
            "chunk_id", "document_id", "asset_id", "document_type",
            "issue_type", "severity", "subsystem", "content", "tags",
            "source_ts", "ingested_at",
        ],
    )
    workspace.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=vs_spec,
    )
    logger.info("Created Vector Search index %s", VS_INDEX)


def _ensure_lakebase_project(workspace: WorkspaceClient) -> str:
    try:
        existing = workspace.postgres.get_project(project_id=LAKEBASE_PROJECT_ID)
        logger.info("Lakebase project %s already exists", existing.name)
        return existing.name
    except NotFound:
        pass

    try:
        operation = workspace.postgres.create_project(
            project=Project(spec=ProjectSpec(display_name=LAKEBASE_DISPLAY_NAME, pg_version="17")),
            project_id=LAKEBASE_PROJECT_ID,
        )
        project = operation.wait()
        logger.info("Created Lakebase project %s", project.name)
        return project.name
    except ResourceAlreadyExists:
        existing = workspace.postgres.get_project(project_id=LAKEBASE_PROJECT_ID)
        return existing.name


def main(force_tables: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    workspace = WorkspaceClient()
    warehouses = list(workspace.warehouses.list())
    running = [
        w for w in warehouses
        if str(getattr(w.state, "value", w.state)).upper() == "RUNNING"
    ]
    if not running:
        raise RuntimeError("No running SQL warehouse found. Start one and rerun.")
    selected_warehouse = running[0].id

    with sql.connect(
        server_hostname=workspace.config.host,
        http_path=f"/sql/1.0/warehouses/{selected_warehouse}",
        credentials_provider=lambda: workspace.config.authenticate,
    ) as conn:
        execute_statements(conn, SCHEMA_DDL)
        assets_exists = _table_exists(conn, f"{CATALOG}.{SCHEMA}.ai_infra_assets")
        chunks_exists = _table_exists(conn, SOURCE_TABLE)
        if force_tables or not (assets_exists and chunks_exists):
            verb = "CREATE OR REPLACE" if force_tables else "CREATE TABLE IF NOT EXISTS"
            # CREATE TABLE IF NOT EXISTS ... AS SELECT isn't supported; fall back per table.
            if not assets_exists or force_tables:
                execute_statements(conn, _tables_ddl("CREATE OR REPLACE").split(";", 1)[0] + ";")
            if not chunks_exists or force_tables:
                parts = _tables_ddl("CREATE OR REPLACE").split(";")
                execute_statements(conn, parts[1].strip() + ";")
            execute_statements(
                conn,
                f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);",
            )
            logger.info("Tables provisioned (force_tables=%s)", force_tables)
        else:
            logger.info("Tables already exist; skipping (use --force-tables to recreate)")

    _ensure_vs_endpoint(workspace)
    _ensure_vs_index(workspace)
    project_name = _ensure_lakebase_project(workspace)

    print(f"Lakebase project: {project_name}")
    print(f"Vector endpoint: {VS_ENDPOINT}")
    print(f"Vector index: {VS_INDEX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-tables", action="store_true",
        help="Recreate the synthetic Delta tables (destructive).",
    )
    args = parser.parse_args()
    main(force_tables=args.force_tables)
