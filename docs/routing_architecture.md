# Routing, Skill Registry & MCP Catalog Architecture

## Overview

The routing layer extends the AI Orchestration framework with skill discovery, intent-based routing, and MCP tool aggregation. All components follow the existing Protocol-based design so any agent runtime can plug in.

```text
User Query
  |
  v
+------------------+     +-------------------+     +-------------------+
| Router           | --> | Skill Registry    | --> | Skill Execution   |
| (Composite)      |     | (discover/get)    |     | (SkillClient)     |
+------------------+     +-------------------+     +-------------------+
  |                         |                          |
  | rules/semantic/LLM     | native + MCP skills      | Vector Search
  |                         |                          | FM Endpoint
  v                         v                          | Lakebase Memory
+------------------+     +-------------------+         | Custom skills
| MLflow Tracing   |     | MCP Catalog       |         |
| (routing spans)  |     | (managed/custom/  |     +-------------------+
+------------------+     |  external servers) |     | RoutedTurnResult  |
                          +-------------------+     +-------------------+
```

## Skill Registry

**Module**: `framework/skill_registry.py`

### Protocols

- `SkillClient` — contract for all skills: `name`, `definition`, `execute(SkillInput) -> SkillResult`, `health()`
- `SkillDefinition` — metadata: name, description, version, tags, input/output schemas, source

### Reference Skills

| Skill | Wraps | Tags |
|-------|-------|------|
| `VectorSearchSkill` | `VectorSearchClient.retrieve()` | retrieval, rag, search |
| `MemoryReadSkill` | `LakebaseMemoryStore.read()` | memory, session |
| `MemoryWriteSkill` | `LakebaseMemoryStore.write()` | memory, write |
| `GenerateSkill` | `FmAgentClient.generate()` | generation, llm |

### Discovery

The registry supports keyword-based discovery via `discover(query, top_k)`. Skills are scored by term overlap between the query and the skill's name, description, and tags.

## MCP Catalog

**Module**: `framework/mcp_catalog_utils.py`

### Server Types

| Type | Description |
|------|-------------|
| **managed** | Databricks-provided MCP servers |
| **custom** | User-deployed MCP servers on Databricks infrastructure |
| **external** | Third-party MCP servers running outside Databricks |

### Operations

- `register_server(config)` — add a server
- `sync_from_mcp_json(path)` — import from `.cursor/mcp.json`
- `list_tools(server_name?)` — discover tools across servers
- `sync_to_skill_registry(registry)` — bridge MCP tools as `source="mcp"` skills

## Router

**Module**: `framework/router.py`

### Router Implementations

| Router | Strategy | Latency | Best For |
|--------|----------|---------|----------|
| `RuleBasedRouter` | Regex pattern matching | ~0ms | Known, deterministic intents |
| `SemanticRouter` | Keyword similarity scoring | ~1ms | Flexible intent matching |
| `LLMRouter` | FM endpoint classification | ~500-2000ms | Complex/ambiguous queries |
| `CompositeRouter` | Cascading tiers | Varies | Production (recommended) |

### CompositeRouter Configuration

```python
router = CompositeRouter()
router.add_tier(RuleBasedRouter(rules=[...]), min_confidence=0.9)
router.add_tier(SemanticRouter(), min_confidence=0.4)
router.add_tier(LLMRouter(fm_client), min_confidence=0.3)
```

Each tier returns a `RoutingDecision` with a confidence score. If confidence >= threshold, routing stops. Otherwise, the next tier is tried.

### Orchestration

`run_routed_turn()` combines routing + execution + memory + tracing:

```
query -> route(query, context) -> registry.get(skill) -> skill.execute(input) -> memory.write -> trace
```

## DSPy Adapter

**Module**: `framework/dspy_adapter.py` (optional dependency)

| Adapter | DSPy Interface | Databricks Backend |
|---------|---------------|-------------------|
| `DatabricksLM` | Language model | FM serving endpoint |
| `DatabricksRetriever` | Retriever | Vector Search |
| `SkillAsTool` | Tool | Any SkillClient |
| `LakebaseMemoryModule` | Module | Lakebase memory |

## LangGraph Adapter

**Module**: `framework/langgraph_adapter.py` (optional dependency)

| Adapter | LangGraph Interface | Databricks Backend |
|---------|-------------------|-------------------|
| `LakebaseCheckpointer` | Checkpoint saver | Lakebase (PostgreSQL) |
| `skill_as_langchain_tool` | BaseTool | Any SkillClient |
| `DatabricksChatModel` | Chat model | FM serving endpoint |

## Evaluation

Routing accuracy is evaluated using `RoutingJudge` which compares actual vs. expected skill selections:

```bash
python scripts/run_routing_eval.py
```

The judge integrates with the existing `JudgeClient` protocol and can be included in MLflow evaluation suites via `make_mlflow_scorer(RoutingJudge())`.

## Decision Guide

| Scenario | Use |
|----------|-----|
| Fixed, known intents | `RuleBasedRouter` only |
| Flexible discovery with low latency | `SemanticRouter` |
| Complex queries, high accuracy needed | `LLMRouter` |
| Production deployment | `CompositeRouter` (rules + semantic + LLM) |
| DSPy program needs Databricks | `dspy_adapter` |
| LangGraph graph needs checkpointing | `langgraph_adapter` with `LakebaseCheckpointer` |
| External agent needs tool catalog | `build_external_skill_catalog_payload()` |
