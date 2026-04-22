"""Publish framework skills as Unity Catalog Functions.

UC Functions created through this module become automatically available as
MCP tools via the managed Functions MCP server at
``<host>/api/2.0/mcp/functions/{catalog}/{schema}`` — so any MCP-capable
agent (including those using ``framework.mcp_client``) can discover and
invoke them through AI Gateway-governed endpoints. This is the coupling
that closes the loop between skills, UC, and MCP.

**Service-mode only.** The generated UC Function body POSTs to a Model
Serving endpoint that hosts the skill; the framework does not host skills
itself. Deploy a skill behind a Model Serving endpoint, then pass its
invocation URL via ``SkillUCBinding.serving_endpoint``.

**Auth in the function body.** The generated Python body reads
``DATABRICKS_TOKEN`` from the function execution environment and uses it
as a Bearer credential. In Databricks, UC Python Functions inherit the
invoker's identity by default for calls to governed services, so this
works when the invoker has Model Serving access. For service-principal
backends, configure a UC secret scope and swap the token source — the
``body_template`` hook (future work) will make that parameterizable.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


@dataclass
class UCColumn:
    """One input parameter of the published UC Function."""

    name: str
    uc_type: str          # "STRING", "DOUBLE", "INT", "ARRAY<STRING>", etc.
    comment: str = ""
    default: str | None = None  # raw SQL literal, e.g. "'hello'" or "NULL"


@dataclass
class SkillUCBinding:
    """Everything needed to publish one skill as one UC Function.

    Bindings are declarative — construction is side-effect-free. Call
    :func:`publish_skill` to create the function in a workspace, or
    :func:`render_function_sql` to inspect the DDL without executing.
    """

    skill_name: str
    catalog: str
    schema: str
    function_name: str
    serving_endpoint: str
    input_columns: list[UCColumn]
    return_type: str = "STRING"
    comment: str = ""
    # Which input column maps to SkillInput.query; the rest become
    # SkillInput.parameters. Must be present in input_columns.
    query_column: str = "query"
    timeout_seconds: int = 30


def _fq_name(binding: SkillUCBinding) -> str:
    return f"`{binding.catalog}`.`{binding.schema}`.`{binding.function_name}`"


def _sql_literal(s: str) -> str:
    """Escape a string for use as a SQL string literal."""
    return "'" + s.replace("'", "''") + "'"


def _render_column_decl(col: UCColumn) -> str:
    parts = [f"`{col.name}` {col.uc_type}"]
    if col.default is not None:
        parts.append(f"DEFAULT {col.default}")
    if col.comment:
        parts.append(f"COMMENT {_sql_literal(col.comment)}")
    return " ".join(parts)


def _validate(binding: SkillUCBinding) -> None:
    if not binding.input_columns:
        raise ValueError(
            f"Binding {binding.skill_name!r}: input_columns cannot be empty",
        )
    names = [c.name for c in binding.input_columns]
    if binding.query_column not in names:
        raise ValueError(
            f"Binding {binding.skill_name!r}: query_column={binding.query_column!r} "
            f"not found in input_columns {names}",
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"Binding {binding.skill_name!r}: duplicate input_column names {names}",
        )


def _render_python_body(binding: SkillUCBinding) -> str:
    """Build the Python body embedded between ``$$ ... $$``.

    The body constructs the Model Serving request, POSTs it, and returns
    the first prediction as a JSON string. Errors propagate as exceptions.
    """
    arg_list = ", ".join(c.name for c in binding.input_columns)
    # dataframe_records payload: query goes in as-is; everything else goes
    # into a nested parameters dict, matching SkillInput.
    param_pairs = ", ".join(
        f'"{c.name}": {c.name}'
        for c in binding.input_columns
        if c.name != binding.query_column
    )
    return textwrap.dedent(f'''
        import json
        import os
        import urllib.request

        def _call({arg_list}):
            payload = json.dumps({{
                "dataframe_records": [{{
                    "query": {binding.query_column},
                    "parameters": {{{param_pairs}}},
                }}]
            }}).encode("utf-8")
            token = os.environ.get("DATABRICKS_TOKEN") or ""
            req = urllib.request.Request(
                "{binding.serving_endpoint}",
                data=payload,
                headers={{
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {{token}}",
                }},
            )
            with urllib.request.urlopen(req, timeout={binding.timeout_seconds}) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            preds = parsed.get("predictions", [parsed])
            return json.dumps(preds[0]) if preds else ""

        return _call({arg_list})
    ''').strip()


def render_function_sql(binding: SkillUCBinding) -> str:
    """Produce the ``CREATE OR REPLACE FUNCTION`` DDL for a binding.

    Pure — does not touch the workspace. Useful for inspection and for
    unit tests that want to assert on the generated SQL shape.
    """
    _validate(binding)
    cols_decl = ",\n    ".join(_render_column_decl(c) for c in binding.input_columns)
    comment_clause = (
        f"\nCOMMENT {_sql_literal(binding.comment)}" if binding.comment else ""
    )
    body = _render_python_body(binding)
    return (
        f"CREATE OR REPLACE FUNCTION {_fq_name(binding)}(\n"
        f"    {cols_decl}\n"
        f")\n"
        f"RETURNS {binding.return_type}\n"
        f"LANGUAGE PYTHON{comment_clause}\n"
        f"AS $$\n{body}\n$$"
    )


def publish_skill(
    binding: SkillUCBinding,
    workspace_client: WorkspaceClient,
    sql_warehouse_id: str,
) -> str:
    """Execute the ``CREATE OR REPLACE FUNCTION`` DDL via a SQL warehouse.

    Returns the fully-qualified function name (``catalog.schema.function``).
    Raises ``RuntimeError`` if the statement does not reach SUCCEEDED state.
    """
    sql = render_function_sql(binding)
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=sql_warehouse_id,
        statement=sql,
    )
    state = getattr(getattr(result, "status", None), "state", None)
    # `state` is an enum with a `.value` or `.name` depending on SDK version;
    # fall back to str() so we don't pin to a specific SDK surface.
    state_label = getattr(state, "value", None) or getattr(state, "name", None) or str(state)
    if state_label not in {"SUCCEEDED", "FINISHED"}:
        error = getattr(getattr(result.status, "error", None), "message", None)
        raise RuntimeError(
            f"Failed to publish UC Function {binding.function_name}: "
            f"state={state_label}; error={error!r}",
        )
    return f"{binding.catalog}.{binding.schema}.{binding.function_name}"


def publish_bindings(
    bindings: list[SkillUCBinding],
    workspace_client: WorkspaceClient,
    sql_warehouse_id: str,
) -> list[str]:
    """Publish many bindings idempotently. Returns the FQ names in order."""
    return [publish_skill(b, workspace_client, sql_warehouse_id) for b in bindings]
