"""Unit tests for framework.uc_function_publisher."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from framework.uc_function_publisher import (
    SkillUCBinding,
    UCColumn,
    publish_bindings,
    publish_skill,
    render_function_sql,
)


def _minimal_binding(**overrides) -> SkillUCBinding:
    base = dict(
        skill_name="search",
        catalog="main",
        schema="agents",
        function_name="search_tool",
        serving_endpoint="https://workspace.cloud.databricks.com/serving-endpoints/search/invocations",
        input_columns=[
            UCColumn(name="query", uc_type="STRING", comment="The search query"),
        ],
        comment="Search UC Functions via the RAG pipeline.",
    )
    base.update(overrides)
    return SkillUCBinding(**base)


# --- SQL rendering -----------------------------------------------------------

def test_render_sql_produces_create_or_replace_for_fq_name():
    sql = render_function_sql(_minimal_binding())
    assert sql.startswith("CREATE OR REPLACE FUNCTION `main`.`agents`.`search_tool`(")


def test_render_sql_includes_column_decls_and_comments():
    sql = render_function_sql(_minimal_binding())
    assert "`query` STRING" in sql
    assert "COMMENT 'The search query'" in sql


def test_render_sql_declares_return_type_and_python_language():
    sql = render_function_sql(_minimal_binding())
    assert "RETURNS STRING" in sql
    assert "LANGUAGE PYTHON" in sql


def test_render_sql_embeds_function_comment():
    sql = render_function_sql(_minimal_binding(comment="Agent tool"))
    assert "COMMENT 'Agent tool'" in sql


def test_render_sql_escapes_single_quotes_in_comments():
    sql = render_function_sql(_minimal_binding(comment="it's a tool"))
    assert "COMMENT 'it''s a tool'" in sql


def test_render_sql_body_hits_serving_endpoint():
    url = "https://ws.cloud.databricks.com/serving-endpoints/my-skill/invocations"
    sql = render_function_sql(_minimal_binding(serving_endpoint=url))
    assert url in sql
    assert 'urllib.request.urlopen' in sql
    assert 'Authorization' in sql


def test_render_sql_maps_query_column_and_parameters():
    binding = _minimal_binding(
        input_columns=[
            UCColumn(name="query", uc_type="STRING"),
            UCColumn(name="top_k", uc_type="INT", default="5"),
            UCColumn(name="filter", uc_type="STRING"),
        ],
    )
    sql = render_function_sql(binding)
    # query column feeds SkillInput.query
    assert '"query": query' in sql
    # the other columns go into SkillInput.parameters
    assert '"top_k": top_k' in sql
    assert '"filter": filter' in sql
    # DEFAULT is rendered in the column decl
    assert "`top_k` INT DEFAULT 5" in sql


def test_render_sql_custom_return_type():
    sql = render_function_sql(_minimal_binding(return_type="ARRAY<STRING>"))
    assert "RETURNS ARRAY<STRING>" in sql


# --- Validation --------------------------------------------------------------

def test_empty_input_columns_raises():
    with pytest.raises(ValueError, match="input_columns cannot be empty"):
        render_function_sql(_minimal_binding(input_columns=[]))


def test_missing_query_column_raises():
    with pytest.raises(ValueError, match="query_column"):
        render_function_sql(_minimal_binding(
            input_columns=[UCColumn(name="not_query", uc_type="STRING")],
        ))


def test_duplicate_column_names_raise():
    with pytest.raises(ValueError, match="duplicate"):
        render_function_sql(_minimal_binding(
            input_columns=[
                UCColumn(name="query", uc_type="STRING"),
                UCColumn(name="query", uc_type="INT"),
            ],
        ))


# --- publish_skill (executes DDL via workspace client) -----------------------

def _ok_wc() -> MagicMock:
    """Build a WorkspaceClient mock whose execute_statement reports SUCCEEDED."""
    wc = MagicMock()
    result = MagicMock()
    result.status.state.value = "SUCCEEDED"
    wc.statement_execution.execute_statement.return_value = result
    return wc


def test_publish_skill_executes_ddl_against_warehouse():
    wc = _ok_wc()
    fq = publish_skill(_minimal_binding(), wc, sql_warehouse_id="wh-1")
    wc.statement_execution.execute_statement.assert_called_once()
    call = wc.statement_execution.execute_statement.call_args
    assert call.kwargs["warehouse_id"] == "wh-1"
    assert call.kwargs["statement"].startswith("CREATE OR REPLACE FUNCTION")
    assert fq == "main.agents.search_tool"


def test_publish_skill_raises_on_failed_state():
    wc = MagicMock()
    result = MagicMock()
    result.status.state.value = "FAILED"
    result.status.error.message = "SQL syntax error"
    wc.statement_execution.execute_statement.return_value = result
    with pytest.raises(RuntimeError, match="Failed to publish.*SQL syntax error"):
        publish_skill(_minimal_binding(), wc, sql_warehouse_id="wh-1")


def test_publish_skill_accepts_finished_state_as_success():
    # Some SDK surfaces use FINISHED instead of SUCCEEDED.
    wc = MagicMock()
    result = MagicMock()
    result.status.state.value = "FINISHED"
    wc.statement_execution.execute_statement.return_value = result
    assert publish_skill(_minimal_binding(), wc, "wh-1") == "main.agents.search_tool"


def test_publish_bindings_publishes_all_in_order():
    wc = _ok_wc()
    b1 = _minimal_binding(function_name="f1")
    b2 = _minimal_binding(function_name="f2")
    names = publish_bindings([b1, b2], wc, sql_warehouse_id="wh-1")
    assert names == ["main.agents.f1", "main.agents.f2"]
    assert wc.statement_execution.execute_statement.call_count == 2


# --- Reference bindings ------------------------------------------------------

def test_vector_search_binding_has_query_and_top_k():
    from framework.reference_uc_bindings import vector_search_binding

    binding = vector_search_binding(
        catalog="main", schema="agents",
        serving_endpoint="https://host/serving-endpoints/vs/invocations",
    )
    names = [c.name for c in binding.input_columns]
    assert names == ["query", "top_k"]
    assert binding.query_column == "query"
    # Binding should render cleanly.
    sql = render_function_sql(binding)
    assert "CREATE OR REPLACE FUNCTION `main`.`agents`.`vector_search`" in sql
    assert "`top_k` INT DEFAULT 5" in sql


def test_generate_binding_has_expected_columns_and_defaults():
    from framework.reference_uc_bindings import generate_binding

    binding = generate_binding(
        catalog="main", schema="agents",
        serving_endpoint="https://host/serving-endpoints/gen/invocations",
    )
    names = [c.name for c in binding.input_columns]
    assert names == ["query", "system_prompt", "temperature"]
    sql = render_function_sql(binding)
    assert "CREATE OR REPLACE FUNCTION `main`.`agents`.`generate`" in sql
    assert "`temperature` DOUBLE DEFAULT 0.2" in sql
