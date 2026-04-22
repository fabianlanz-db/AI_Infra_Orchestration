"""Concrete MCPAuth strategies backed by Databricks workspace credentials.

Three reference implementations satisfy the ``framework.mcp_client.MCPAuth``
Protocol:

* ``WorkspaceClientAuth`` — wraps a ``databricks.sdk.WorkspaceClient`` and
  auto-detects whether the underlying auth is U2M (user) or M2M (service
  principal) for accurate ``mcp.auth_mode`` tagging. This is the primary path
  for both in-workspace agents (Pattern B) and external agents with service
  principal credentials (Pattern A).
* ``PATAuth`` — static Bearer token. Simple; useful for local dev and
  integration tests. **Not supported for custom MCP servers** per the
  Databricks docs (connect-external-services.md → "Personal access tokens
  are only supported for managed and external MCP servers").
* ``auto_select_auth`` — picks a strategy from ambient environment.

The compatibility check ``ensure_auth_compatible_with_server`` enforces the
custom-MCP + PAT constraint at registration time so misconfiguration fails
loudly rather than at the first tool invocation.
"""
from __future__ import annotations

import os

from databricks.sdk import WorkspaceClient

from framework.mcp_client import MCPAuth


class WorkspaceClientAuth:
    """MCPAuth backed by a ``WorkspaceClient``.

    Delegates token lifecycle to the SDK — OAuth U2M refresh, SP M2M token
    exchange, and PAT passthrough are all handled upstream. This class just
    pulls the current Bearer token out of ``config.authenticate()`` on each
    request, so refresh happens naturally.
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        mode: str | None = None,
    ) -> None:
        self._wc = workspace_client
        self._mode_override = mode

    def bearer_token(self) -> str:
        headers = self._wc.config.authenticate()
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError(
                "WorkspaceClient returned a non-Bearer Authorization header; "
                "MCP requires Bearer tokens. Check your Databricks SDK auth config."
            )
        return auth_header[len("Bearer "):]

    def mode(self) -> str:
        if self._mode_override:
            return self._mode_override
        auth_type = (self._wc.config.auth_type or "").lower()
        # Databricks SDK auth_type values map to our three labels:
        #   "pat"                                   -> pat
        #   "oauth-m2m" / "*client-secret" / "*client_credentials" -> m2m
        #   everything else (u2m, cli, notebook, external-browser) -> u2m
        if auth_type == "pat":
            return "pat"
        # Databricks SDK uses kebab-case auth_type values. M2M variants all
        # contain either "m2m" or "client-secret".
        if "m2m" in auth_type or "client-secret" in auth_type:
            return "m2m"
        return "u2m"


class PATAuth:
    """Static Personal Access Token bearer auth.

    Simpler than OAuth but has real limits: Databricks custom MCP servers
    (hosted on Apps) do NOT accept PAT — use ``WorkspaceClientAuth`` there.
    """

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("PAT cannot be empty")
        self._token = token

    def bearer_token(self) -> str:
        return self._token

    def mode(self) -> str:
        return "pat"


def auto_select_auth(workspace_client: WorkspaceClient | None = None) -> MCPAuth:
    """Pick an auth strategy from the ambient environment.

    Priority:

    1. Caller-supplied ``workspace_client`` (explicit wins).
    2. ``DATABRICKS_CLIENT_ID`` + ``DATABRICKS_CLIENT_SECRET`` → WC with M2M.
    3. ``DATABRICKS_TOKEN`` → ``PATAuth``.
    4. Default ``WorkspaceClient()`` — picks up CLI / config file / notebook.
    """
    if workspace_client is not None:
        return WorkspaceClientAuth(workspace_client)
    if os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"):
        # WorkspaceClient() auto-selects M2M when both env vars are present.
        return WorkspaceClientAuth(WorkspaceClient(), mode="m2m")
    if os.environ.get("DATABRICKS_TOKEN"):
        return PATAuth(os.environ["DATABRICKS_TOKEN"])
    return WorkspaceClientAuth(WorkspaceClient())


def ensure_auth_compatible_with_server(auth_mode: str, server_type: str) -> None:
    """Fail fast if the auth mode is incompatible with the server type.

    Databricks custom MCP servers (hosted on Apps) require OAuth; PAT is not
    accepted. Managed and external MCP servers accept either.
    """
    if server_type == "custom" and auth_mode == "pat":
        raise ValueError(
            "Custom MCP servers (Databricks Apps) do not support PAT "
            "authentication. Use WorkspaceClientAuth (U2M or M2M OAuth) or "
            "supply a BYO session_factory to DatabricksMCPClient."
        )
