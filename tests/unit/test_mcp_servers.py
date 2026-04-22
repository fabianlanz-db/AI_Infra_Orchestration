"""Unit tests for framework.mcp_servers factory helpers."""
from __future__ import annotations

import pytest

from framework.mcp_servers import (
    custom_server,
    external_server,
    managed_functions_server,
    managed_genie_server,
    managed_vector_search_server,
)

HOST = "https://my-workspace.cloud.databricks.com"


# --- URL construction --------------------------------------------------------

def test_managed_functions_url_and_type():
    cfg = managed_functions_server("main", "analytics", HOST)
    assert cfg.url == f"{HOST}/api/2.0/mcp/functions/main/analytics"
    assert cfg.server_type == "managed"
    assert cfg.metadata == {"uc_catalog": "main", "uc_schema": "analytics", "kind": "functions"}


def test_managed_vector_search_url():
    cfg = managed_vector_search_server("main", "rag", HOST)
    assert cfg.url == f"{HOST}/api/2.0/mcp/vector-search/main/rag"
    assert cfg.metadata["kind"] == "vector-search"


def test_managed_genie_url():
    cfg = managed_genie_server("space-abc-123", HOST)
    assert cfg.url == f"{HOST}/api/2.0/mcp/genie/space-abc-123"
    assert cfg.metadata["genie_space_id"] == "space-abc-123"


@pytest.mark.parametrize("raw_host,expected_base", [
    ("my-workspace.cloud.databricks.com", "https://my-workspace.cloud.databricks.com"),
    ("https://my-workspace.cloud.databricks.com/", "https://my-workspace.cloud.databricks.com"),
    ("http://localhost:8080", "http://localhost:8080"),
])
def test_host_normalization(raw_host: str, expected_base: str):
    cfg = managed_functions_server("c", "s", raw_host)
    assert cfg.url == f"{expected_base}/api/2.0/mcp/functions/c/s"


def test_name_defaults_and_overrides():
    default = managed_functions_server("cat", "sch", HOST)
    assert default.name == "uc-functions-cat-sch"
    custom = managed_functions_server("cat", "sch", HOST, name="custom-name")
    assert custom.name == "custom-name"


# --- External + custom -------------------------------------------------------

def test_external_server_uses_explicit_url():
    cfg = external_server("vendor-x", "https://vendor.example.com/mcp")
    assert cfg.server_type == "external"
    assert cfg.url == "https://vendor.example.com/mcp"


def test_custom_server_uses_app_url():
    cfg = custom_server("my-app", "https://my-app.cloud.databricksapps.com/mcp")
    assert cfg.server_type == "custom"
    assert cfg.url == "https://my-app.cloud.databricksapps.com/mcp"


def test_metadata_passthrough():
    meta = {"team": "data-platform", "contact": "team@x"}
    assert external_server("x", "https://x/mcp", metadata=meta).metadata == meta
    assert custom_server("x", "https://x/mcp", metadata=meta).metadata == meta
