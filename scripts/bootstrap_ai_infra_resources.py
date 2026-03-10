"""
Bootstrap Databricks resources using existing external-agent resource names.

Creates/updates:
- UC schema and synthetic Delta tables in fl_demos.asml_external_agent_demo
- Vector Search endpoint + Delta Sync index
- Lakebase Autoscaling project
"""

from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, ProjectSpec
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)


CATALOG = "fl_demos"
SCHEMA = "asml_external_agent_demo"
VS_ENDPOINT = "asml_external_agent_vs_ep"
VS_INDEX = f"{CATALOG}.{SCHEMA}.asml_kb_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.asml_kb_chunks"
LAKEBASE_PROJECT_ID = "asml-external-agent-db"


DDL_AND_DATA = f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA};

CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.asml_assets USING DELTA AS
SELECT
  concat('ASSET-', lpad(cast(id + 1 as string), 6, '0')) AS asset_id,
  element_at(array('EUVScanner','DUVScanner','MetrologySystem','EtchModule','LithographyTrack'), cast(rand() * 5 as int) + 1) AS asset_type,
  element_at(array('Optics','WaferHandling','VacuumSystem','ThermalControl','LaserSubsystem'), cast(rand() * 5 as int) + 1) AS subsystem,
  element_at(array('Veldhoven','Berlin','SanDiego','Hsinchu','Phoenix'), cast(rand() * 5 as int) + 1) AS facility,
  element_at(array('active','maintenance','degraded'), cast(rand() * 3 as int) + 1) AS asset_state,
  date_sub(current_date(), cast(rand() * 2000 as int)) AS installed_date,
  current_timestamp() AS ingested_at
FROM range(500);

CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.asml_kb_chunks USING DELTA AS
SELECT
  concat('CH-', lpad(cast(id + 1 as string), 8, '0')) AS chunk_id,
  concat('DOC-', lpad(cast(cast(rand() * 900 as int) + 1 as string), 6, '0')) AS document_id,
  concat('ASSET-', lpad(cast(cast(rand() * 500 as int) + 1 as string), 6, '0')) AS asset_id,
  element_at(array('runbook','incident_report','sop','engineering_note','root_cause_analysis'), cast(rand() * 5 as int) + 1) AS document_type,
  element_at(array('alignment_drift','vacuum_alert','laser_instability','cooling_warning','sensor_fault'), cast(rand() * 5 as int) + 1) AS issue_type,
  element_at(array('low','medium','high','critical'), cast(rand() * 4 as int) + 1) AS severity,
  element_at(array('Optics','WaferHandling','VacuumSystem','ThermalControl','LaserSubsystem'), cast(rand() * 5 as int) + 1) AS subsystem,
  concat('ASML knowledge chunk ', cast(id as string), ' describing diagnostics, probable root cause, stepwise mitigation, safety constraints, and part replacement guidance for high-precision lithography operations.') AS content,
  concat_ws(',', 'asml', 'external-agent', 'ops-demo') AS tags,
  timestampadd(HOUR, -cast(rand() * 24 * 365 as int), current_timestamp()) AS source_ts,
  current_timestamp() AS ingested_at
FROM range(8000);
"""


def execute_statements(conn: sql.Connection, sql_text: str) -> None:
    with conn.cursor() as cur:
        for stmt in [s.strip() for s in sql_text.split(";") if s.strip()]:
            cur.execute(stmt)


def main() -> None:
    workspace = WorkspaceClient()
    warehouse_id = workspace.warehouses.get_workspace_warehouse_config().data_access_config[0].key if False else None
    warehouses = list(workspace.warehouses.list())
    running = [w for w in warehouses if str(w.state) == "RUNNING"]
    if not running:
        raise RuntimeError("No running SQL warehouse found. Start one and rerun.")
    selected_warehouse = running[0].id

    with sql.connect(
        server_hostname=workspace.config.host,
        http_path=f"/sql/1.0/warehouses/{selected_warehouse}",
        credentials_provider=lambda: workspace.config.authenticate,
    ) as conn:
        execute_statements(conn, DDL_AND_DATA)
        execute_statements(
            conn,
            f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);",
        )

    workspace.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
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
            "chunk_id",
            "document_id",
            "asset_id",
            "document_type",
            "issue_type",
            "severity",
            "subsystem",
            "content",
            "tags",
            "source_ts",
            "ingested_at",
        ],
    )
    workspace.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=vs_spec,
    )

    operation = workspace.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name="ASML External Agent DB", pg_version="17")),
        project_id=LAKEBASE_PROJECT_ID,
    )
    project = operation.wait()
    print(f"Created/updated lakebase project: {project.name}")
    print(f"Vector endpoint: {VS_ENDPOINT}")
    print(f"Vector index: {VS_INDEX}")


if __name__ == "__main__":
    main()
