"""Reference ``SkillUCBinding`` values for the framework's canonical skills.

These are worked examples showing how to publish the reference skills
from :mod:`framework.reference_skills` as UC Functions. ``MemoryWriteSkill``
is intentionally omitted: it writes to session memory and is too
permissions-sensitive to expose as a UC Function callable by arbitrary
principals. ``MemoryReadSkill`` is also omitted for the same reason; if
you need per-session history surfaced to MCP clients, publish a scoped
variant that restricts to the caller's own session via UC grants.

Usage::

    from databricks.sdk import WorkspaceClient
    from framework.reference_uc_bindings import vector_search_binding
    from framework.uc_function_publisher import publish_skill

    binding = vector_search_binding(
        catalog="main",
        schema="agents",
        serving_endpoint="https://<host>/serving-endpoints/vs-skill/invocations",
    )
    publish_skill(binding, WorkspaceClient(), sql_warehouse_id="wh-xyz")
"""
from __future__ import annotations

from framework.uc_function_publisher import SkillUCBinding, UCColumn


def vector_search_binding(
    catalog: str,
    schema: str,
    serving_endpoint: str,
    function_name: str = "vector_search",
) -> SkillUCBinding:
    """UC Function binding for :class:`~framework.reference_skills.VectorSearchSkill`.

    Exposes two parameters matching the skill's input contract: the query
    text, and an optional ``top_k`` for number of results.
    """
    return SkillUCBinding(
        skill_name="vector-search",
        catalog=catalog,
        schema=schema,
        function_name=function_name,
        serving_endpoint=serving_endpoint,
        input_columns=[
            UCColumn(
                name="query", uc_type="STRING",
                comment="Text to retrieve documents for.",
            ),
            UCColumn(
                name="top_k", uc_type="INT", default="5",
                comment="Number of documents to return.",
            ),
        ],
        return_type="STRING",
        comment=(
            "Retrieve documents from Databricks Vector Search via the "
            "VectorSearchSkill endpoint. Governed by Unity Catalog."
        ),
    )


def generate_binding(
    catalog: str,
    schema: str,
    serving_endpoint: str,
    function_name: str = "generate",
) -> SkillUCBinding:
    """UC Function binding for :class:`~framework.reference_skills.GenerateSkill`.

    Exposes the prompt (``query``), an optional ``system_prompt`` override,
    and ``temperature``.
    """
    return SkillUCBinding(
        skill_name="generate",
        catalog=catalog,
        schema=schema,
        function_name=function_name,
        serving_endpoint=serving_endpoint,
        input_columns=[
            UCColumn(
                name="query", uc_type="STRING",
                comment="User prompt to send to the LLM.",
            ),
            UCColumn(
                name="system_prompt", uc_type="STRING",
                default="'You are a helpful assistant.'",
                comment="System prompt override.",
            ),
            UCColumn(
                name="temperature", uc_type="DOUBLE", default="0.2",
                comment="Sampling temperature.",
            ),
        ],
        return_type="STRING",
        comment=(
            "Generate text via the GenerateSkill Model Serving endpoint. "
            "Governed by Unity Catalog."
        ),
    )
