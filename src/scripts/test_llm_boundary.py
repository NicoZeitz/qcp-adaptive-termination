"""
Test the LLM boundary finder for a single query.

Loads agents and query from an input JSON, builds column embeddings, ranks them by
similarity to the query, prints the ranked list, then fires the boundary-finder
LLM call. Set log_llm_requests: true in config to also print the raw curl.

Usage:
    # From repo root:
    python3 -m scripts.test_llm_boundary \\
        --input /path/to/input.json \\
        --data_dir /path/to/dataset \\
        --config config.yaml \\
        --cache_dir .cache

    # Or directly (adds src/ to path automatically):
    python3 src/scripts/test_llm_boundary.py \\
        --input /path/to/input.json \\
        --data_dir /path/to/dataset \\
        --config config.yaml \\
        --cache_dir .cache
"""

import sys
from pathlib import Path

# Allow running directly from any working directory (src/ and repo root both need to be in path)
_src = Path(__file__).parent.parent
_repo = _src.parent
for _p in (str(_src), str(_repo)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import logging
from typing import cast

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.models.agent import AgentRole
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from pydantic import BaseModel
from scripts.utils import ScriptContext, run_script


class BoundaryResult(BaseModel):
    reasoning: str
    relevant_agent_names: list[str]


async def _run(ctx: ScriptContext) -> None:
    cqa = cast(CollaborativeQueryAnswering, ctx.data)
    data_agents: list[DataAgent] = cqa.data_agents
    query: str = cqa.query

    print(f"Query: {query}")
    print(f"Agents: {len(data_agents)}\n")

    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    col_desc_map: dict[tuple[str, str, str], str] = {}
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                entry = (agent.name, table_name, col_name)
                desc = f"{table_desc.natural_language_description} {col_desc.natural_language_description}"
                col_entries.append(entry)
                col_descriptions.append(desc)
                col_desc_map[entry] = desc

    col_embeddings = await embedding.create(col_descriptions)
    vector_index = embedding.VectorIndex.build(list(zip(col_entries, col_embeddings)))

    agent_best: dict[str, tuple[float, str, str, str]] = {}
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best or dist < agent_best[agent_name][0]:
            agent_best[agent_name] = (dist, table_name, col_name, col_desc_map[(agent_name, table_name, col_name)])

    sorted_agents = sorted(agent_best.items(), key=lambda x: x[1][0])

    print("Ranked columns (ascending L2):")
    ranked_lines = []
    for rank, (agent_name, (dist, table_name, col_name, desc)) in enumerate(sorted_agents, start=1):
        print(f"[{rank}] {agent_name} | {desc} | L2={dist:.4f}")
        ranked_lines.append(f"[{rank}] {agent_name} | table: {table_name} · column: {col_name} | L2={dist:.4f}")
    print()

    result = await runner.run_llm(
        role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
        prompt_parameters={
            "query": query,
            "ranked_columns": "\n".join(ranked_lines),
        },
        output_type=BoundaryResult,
    )

    print("LLM result:")
    print(f"  relevant_agent_names: {result.relevant_agent_names}")
    print(f"  reasoning: {result.reasoning}")

    ground_truth_names = set((ctx.extra or {}).get("relevant_data_agents", []))
    if ground_truth_names:
        selected = set(result.relevant_agent_names)
        tp = len(ground_truth_names & selected)
        fp = len(selected - ground_truth_names)
        fn = len(ground_truth_names - selected)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print()
        print(f"Ground truth: {sorted(ground_truth_names)}")
        print(f"Missed: {sorted(ground_truth_names - selected)}")
        print(f"Extra:  {sorted(selected - ground_truth_names)}")
        print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_script(
        description="Test LLM boundary finder for a single query",
        run_fn=_run,
    )
