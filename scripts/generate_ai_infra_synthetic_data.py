"""
Generate synthetic AI infrastructure demo data in Unity Catalog.

Required environment variables (no defaults — fails fast if missing):
  DEMO_CATALOG, DEMO_SCHEMA
"""

import os
import sys

from databricks import sql
from databricks.sdk import WorkspaceClient


_missing = [n for n in ("DEMO_CATALOG", "DEMO_SCHEMA") if not os.environ.get(n)]
if _missing:
    sys.exit(
        "Missing required environment variables: "
        + ", ".join(_missing)
        + ". Set DEMO_CATALOG and DEMO_SCHEMA before running."
    )

CATALOG = os.environ["DEMO_CATALOG"]
SCHEMA = os.environ["DEMO_SCHEMA"]


SQL_CONTENT = f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA};

CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.ai_infra_maintenance_events USING DELTA AS
SELECT
  concat('EVT-', lpad(cast(id + 1 as string), 7, '0')) AS event_id,
  concat('ASSET-', lpad(cast(cast(rand() * 500 as int) + 1 as string), 6, '0')) AS asset_id,
  element_at(array('alignment_drift','vacuum_alert','laser_instability','cooling_warning','sensor_fault'), cast(rand() * 5 as int) + 1) AS issue_type,
  element_at(array('low','medium','high','critical'), cast(rand() * 4 as int) + 1) AS severity,
  element_at(array('open','triaged','resolved'), cast(rand() * 3 as int) + 1) AS status,
  timestampadd(HOUR, -cast(rand() * 24 * 180 as int), current_timestamp()) AS event_ts,
  concat('Synthetic event ', cast(id as string), ' generated for AI infrastructure demo.') AS event_summary
FROM range(3000);

CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.ai_infra_parts_inventory USING DELTA AS
SELECT
  concat('PART-', lpad(cast(id + 1 as string), 6, '0')) AS part_id,
  element_at(array('Optics','VacuumSystem','LaserSubsystem','WaferHandling','ThermalControl'), cast(rand() * 5 as int) + 1) AS subsystem,
  element_at(array('lens','seal','pump','sensor','coolant','mirror','controller'), cast(rand() * 7 as int) + 1) AS part_category,
  cast(rand() * 180 as int) AS stock_qty,
  cast(100 + rand() * 25000 as decimal(18,2)) AS unit_cost_eur,
  element_at(array('in_stock','low_stock','backorder'), cast(rand() * 3 as int) + 1) AS supply_status,
  current_timestamp() AS updated_at
FROM range(1200);
"""


def main() -> None:
    workspace = WorkspaceClient()
    warehouses = list(workspace.warehouses.list())
    running = [
        w
        for w in warehouses
        if str(getattr(w.state, "value", w.state)).upper() == "RUNNING"
    ]
    if not running:
        raise RuntimeError("No running SQL warehouse found. Start one and rerun.")
    warehouse_id = running[0].id

    with sql.connect(
        server_hostname=workspace.config.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: workspace.config.authenticate,
    ) as conn, conn.cursor() as cur:
        for stmt in [s.strip() for s in SQL_CONTENT.split(";") if s.strip()]:
            cur.execute(stmt)

        cur.execute(
            f"""
            SELECT
              (SELECT count(*) FROM {CATALOG}.{SCHEMA}.ai_infra_maintenance_events) AS maintenance_events,
              (SELECT count(*) FROM {CATALOG}.{SCHEMA}.ai_infra_parts_inventory) AS parts_inventory
            """
        )
        print(cur.fetchall())


if __name__ == "__main__":
    main()
