"""Unit tests for framework.mcp_auth."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from framework.mcp_auth import (
    PATAuth,
    WorkspaceClientAuth,
    auto_select_auth,
    ensure_auth_compatible_with_server,
)


def _make_wc(auth_type: str, bearer_token: str = "tok") -> MagicMock:
    """Construct a WorkspaceClient mock with the given auth_type + token."""
    wc = MagicMock()
    wc.config.auth_type = auth_type
    wc.config.authenticate.return_value = {"Authorization": f"Bearer {bearer_token}"}
    return wc


# --- PATAuth -----------------------------------------------------------------

def test_pat_auth_rejects_empty_token():
    with pytest.raises(ValueError, match="PAT cannot be empty"):
        PATAuth("")


def test_pat_auth_returns_token_and_mode():
    auth = PATAuth("t-123")
    assert auth.bearer_token() == "t-123"
    assert auth.mode() == "pat"


# --- WorkspaceClientAuth -----------------------------------------------------

def test_workspace_auth_extracts_bearer_from_headers():
    wc = _make_wc("databricks-cli", bearer_token="abc")
    assert WorkspaceClientAuth(wc).bearer_token() == "abc"


def test_workspace_auth_raises_on_non_bearer():
    wc = _make_wc("databricks-cli")
    wc.config.authenticate.return_value = {"Authorization": "Basic base64=="}
    with pytest.raises(RuntimeError, match="non-Bearer"):
        WorkspaceClientAuth(wc).bearer_token()


@pytest.mark.parametrize("auth_type,expected", [
    ("pat", "pat"),
    ("oauth-m2m", "m2m"),
    ("azure-client-secret", "m2m"),
    ("databricks-cli", "u2m"),
    ("external-browser", "u2m"),
    ("notebook", "u2m"),
    ("", "u2m"),  # fallback
])
def test_workspace_auth_mode_classification(auth_type: str, expected: str):
    wc = _make_wc(auth_type)
    assert WorkspaceClientAuth(wc).mode() == expected


def test_workspace_auth_mode_override():
    wc = _make_wc("databricks-cli")
    assert WorkspaceClientAuth(wc, mode="m2m").mode() == "m2m"


# --- auto_select_auth --------------------------------------------------------

def test_auto_select_explicit_workspace_client_wins(monkeypatch):
    # Even with SP env vars set, an explicit WC should be used as-is.
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "cid")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "csec")
    wc = _make_wc("databricks-cli")
    auth = auto_select_auth(workspace_client=wc)
    assert isinstance(auth, WorkspaceClientAuth)
    # No mode override → mode determined by wc auth_type
    assert auth.mode() == "u2m"


def test_auto_select_service_principal_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "cid")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "csec")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    with patch("framework.mcp_auth.WorkspaceClient", return_value=_make_wc("oauth-m2m")):
        auth = auto_select_auth()
    assert isinstance(auth, WorkspaceClientAuth)
    # Mode override passed through → "m2m"
    assert auth.mode() == "m2m"


def test_auto_select_pat_env(monkeypatch):
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "pat-xyz")
    auth = auto_select_auth()
    assert isinstance(auth, PATAuth)
    assert auth.bearer_token() == "pat-xyz"


def test_auto_select_falls_back_to_workspace_client(monkeypatch):
    for k in ["DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET", "DATABRICKS_TOKEN"]:
        monkeypatch.delenv(k, raising=False)
    with patch("framework.mcp_auth.WorkspaceClient", return_value=_make_wc("databricks-cli")):
        auth = auto_select_auth()
    assert isinstance(auth, WorkspaceClientAuth)


# --- ensure_auth_compatible_with_server --------------------------------------

def test_custom_server_rejects_pat():
    with pytest.raises(ValueError, match="(?i)custom mcp"):
        ensure_auth_compatible_with_server(auth_mode="pat", server_type="custom")


@pytest.mark.parametrize("server_type", ["managed", "external"])
def test_pat_allowed_for_managed_and_external(server_type: str):
    ensure_auth_compatible_with_server(auth_mode="pat", server_type=server_type)


@pytest.mark.parametrize("auth_mode", ["u2m", "m2m", "byo"])
def test_oauth_and_byo_allowed_for_custom(auth_mode: str):
    ensure_auth_compatible_with_server(auth_mode=auth_mode, server_type="custom")
