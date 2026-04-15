"""
Bootstrap the skill catalog with reference skills and MCP servers.

Registers reference skills (Vector Search, Memory, Generate) in the
SkillRegistry, imports MCP server configs from .cursor/mcp.json, and
bridges MCP tools into the registry for unified discovery.

Usage:
    python scripts/bootstrap_skill_catalog.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.mcp_catalog_utils import MCPCatalogClient
from framework.reference_skills import (
    GenerateSkill,
    MemoryReadSkill,
    MemoryWriteSkill,
    VectorSearchSkill,
)
from framework.skill_registry import SkillRegistry
from framework.vector_search_utils import VectorSearchClient


def bootstrap(mcp_json_path: str = ".cursor/mcp.json") -> dict:
    """Bootstrap the full skill catalog.

    Returns a summary dict with counts and status.
    """
    registry = SkillRegistry()
    errors: list[str] = []

    # -- Register reference skills (skip if env not configured) ----------------
    skill_count = 0
    try:
        vs_client = VectorSearchClient()
        registry.register(VectorSearchSkill(vs_client))
        skill_count += 1
        print("[OK] Registered VectorSearchSkill")
    except Exception as exc:
        errors.append(f"VectorSearchSkill: {exc}")
        print(f"[SKIP] VectorSearchSkill: {exc}")

    try:
        memory = LakebaseMemoryStore()
        registry.register(MemoryReadSkill(memory))
        registry.register(MemoryWriteSkill(memory))
        skill_count += 2
        print("[OK] Registered MemoryReadSkill + MemoryWriteSkill")
    except Exception as exc:
        errors.append(f"MemorySkills: {exc}")
        print(f"[SKIP] MemorySkills: {exc}")

    try:
        fm = FmAgentClient()
        registry.register(GenerateSkill(fm))
        skill_count += 1
        print("[OK] Registered GenerateSkill")
    except Exception as exc:
        errors.append(f"GenerateSkill: {exc}")
        print(f"[SKIP] GenerateSkill: {exc}")

    # -- Import MCP servers ----------------------------------------------------
    catalog = MCPCatalogClient()
    mcp_count = catalog.sync_from_mcp_json(mcp_json_path)
    print(f"[OK] Imported {mcp_count} MCP server(s) from {mcp_json_path}")

    # -- Bridge MCP tools into skill registry ----------------------------------
    bridged = catalog.sync_to_skill_registry(registry)
    print(f"[OK] Bridged {bridged} MCP tool(s) into skill registry")

    # -- Summary ---------------------------------------------------------------
    ok, msg = registry.health()
    catalog_ok, catalog_msg = catalog.health()

    summary = {
        "registry_status": msg,
        "catalog_status": catalog_msg,
        "reference_skills": skill_count,
        "mcp_servers": mcp_count,
        "mcp_tools_bridged": bridged,
        "total_skills": len(registry.list_skills()),
        "errors": errors,
    }
    print(f"\nCatalog: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    bootstrap()
