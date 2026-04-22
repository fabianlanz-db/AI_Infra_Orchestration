"""Factory helpers for ``MCPServerConfig``, one per Databricks MCP type.

Per the Databricks docs (``learn.microsoft.com/azure/databricks/generative-ai/mcp/``
and ``.../connect-external-services``), Databricks hosts three kinds of MCP
servers — managed, external (via UC connections), custom (via Apps) — and
managed servers expose tools at documented URL paths. This module encodes
those paths so callers don't hand-assemble URLs.

There is no documented API today to enumerate MCP servers across a
workspace (Q10). Callers register servers deterministically by passing in
the catalog/schema/Genie space IDs they want exposed.
"""
from __future__ import annotations

from typing import Any

from framework.mcp_catalog_utils import MCPServerConfig


def _managed_url(workspace_host: str, path: str) -> str:
    """Build a managed MCP URL. Prepends https:// and strips trailing slashes."""
    host = workspace_host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return f"{host}/api/2.0/mcp/{path.lstrip('/')}"


def managed_functions_server(
    catalog: str,
    schema: str,
    workspace_host: str,
    name: str | None = None,
) -> MCPServerConfig:
    """UC Functions in ``catalog.schema`` exposed as MCP tools.

    URL: ``<host>/api/2.0/mcp/functions/{catalog}/{schema}``.

    This is the coupling that makes skills-published-as-UC-Functions
    (Step 7) discoverable as MCP tools by any client — Databricks exposes
    UC Functions through this managed endpoint automatically.
    """
    return MCPServerConfig(
        name=name or f"uc-functions-{catalog}-{schema}",
        server_type="managed",
        url=_managed_url(workspace_host, f"functions/{catalog}/{schema}"),
        metadata={"uc_catalog": catalog, "uc_schema": schema, "kind": "functions"},
    )


def managed_vector_search_server(
    catalog: str,
    schema: str,
    workspace_host: str,
    name: str | None = None,
) -> MCPServerConfig:
    """Vector Search indices in ``catalog.schema`` exposed as MCP tools.

    URL: ``<host>/api/2.0/mcp/vector-search/{catalog}/{schema}`` (pattern
    inferred from the managed-MCP URL conventions; verify against the
    Managed MCP docs if discovery returns empty for your workspace).
    """
    return MCPServerConfig(
        name=name or f"uc-vector-search-{catalog}-{schema}",
        server_type="managed",
        url=_managed_url(workspace_host, f"vector-search/{catalog}/{schema}"),
        metadata={"uc_catalog": catalog, "uc_schema": schema, "kind": "vector-search"},
    )


def managed_genie_server(
    genie_space_id: str,
    workspace_host: str,
    name: str | None = None,
) -> MCPServerConfig:
    """A Genie space exposed as MCP tools.

    URL: ``<host>/api/2.0/mcp/genie/{genie_space_id}``.
    """
    return MCPServerConfig(
        name=name or f"genie-{genie_space_id}",
        server_type="managed",
        url=_managed_url(workspace_host, f"genie/{genie_space_id}"),
        metadata={"genie_space_id": genie_space_id, "kind": "genie"},
    )


def external_server(
    name: str,
    url: str,
    metadata: dict[str, Any] | None = None,
) -> MCPServerConfig:
    """External MCP server registered via a UC managed connection.

    The framework does not enumerate UC connections (no workspace API for
    that today); pass the URL explicitly. AI Gateway still governs these
    calls via UC permissions on the connection.
    """
    return MCPServerConfig(
        name=name,
        server_type="external",
        url=url,
        metadata=metadata or {},
    )


def custom_server(
    name: str,
    app_url: str,
    metadata: dict[str, Any] | None = None,
) -> MCPServerConfig:
    """Custom MCP server hosted as a Databricks App.

    ``app_url`` is the deployed App URL. Custom MCP servers require OAuth
    (U2M or M2M) — PAT is NOT supported per the Databricks docs
    (``connect-external-services.md``).
    """
    return MCPServerConfig(
        name=name,
        server_type="custom",
        url=app_url,
        metadata=metadata or {},
    )
