"""
Experiment: can an LLM find the relevance boundary in a ranked column list?

Reads existing embedding-pre-filter logs (stage_1_output_embedding_pre_filter.json)
and the original input files (for ground-truth relevant_data_agents). For each query,
reconstructs the ranked column list from l2_scores + best_columns and asks the LLM
to identify which agents are relevant — without seeing any schema, just table+column
descriptions embedded in the column text.

Prints precision / recall / F1 per query and aggregate, compared to ground truth.

Usage:
    python3 -m scripts.eval_llm_boundary \\
        --log_dir /mnt/labstore/fwolf/.logs/.../eval_stage1_recall \\
        --input_dir /mnt/labstore/nzeitz/dataset/.../input \\
        --config config.yaml
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models.agent import AgentRole
from qcp_adaptive_termination.observability import context as obs_context
from qcp_adaptive_termination.pipeline import load_config
from pydantic import BaseModel


class BoundaryResult(BaseModel):
    reasoning: str
    relevant_agent_names: list[str]


def _find_latest_json(query_log_dir: Path, filename: str) -> dict | None:
    candidates = sorted(query_log_dir.glob(f"*/{filename}"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def _find_input_file(query_log_dir: Path, input_dir: Path | None) -> Path | None:
    slug = query_log_dir.name
    if input_dir:
        p = input_dir / f"{slug}.json"
        if p.exists():
            return p
    candidates = sorted(query_log_dir.glob(f"*/{slug}.json"))
    return candidates[-1] if candidates else None


def _compute_metrics(relevant: list[str], selected: list[str]) -> tuple[float, float, float, int, int, int]:
    tp = len(set(relevant) & set(selected))
    fp = len(set(selected) - set(relevant))
    fn = len(set(relevant) - set(selected))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _build_ranked_columns(
    l2_scores: dict[str, float],
    best_columns: dict[str, str],
) -> str:
    sorted_agents = sorted(l2_scores.items(), key=lambda x: x[1])
    lines = []
    for rank, (agent_name, l2) in enumerate(sorted_agents, start=1):
        table_col = best_columns.get(agent_name, "unknown.unknown")
        parts = table_col.split(".", 1)
        table = parts[0] if len(parts) == 2 else table_col
        col = parts[1] if len(parts) == 2 else table_col
        lines.append(f"[{rank}] {agent_name} | table: {table} · column: {col} | L2={l2:.4f}")
    return "\n".join(lines)


async def _process_query(
    query_log_dir: Path,
    input_dir: Path | None,
) -> dict | None:
    slug = query_log_dir.name

    input_file = _find_input_file(query_log_dir, input_dir)
    if input_file is None:
        logging.warning("%s: input file not found — skipping", slug)
        return None

    with open(input_file) as f:
        input_data = json.load(f)

    relevant: list[str] = input_data.get("extra", {}).get("relevant_data_agents", [])
    if not relevant:
        logging.info("%s: no relevant_data_agents in extra — skipping", slug)
        return None

    emb = _find_latest_json(query_log_dir, "stage_1_output_embedding_pre_filter.json")
    if emb is None:
        logging.warning("%s: no embedding pre-filter log found — skipping", slug)
        return None

    query: str = emb.get("query", "")
    l2_scores: dict[str, float] = emb.get("l2_scores", {})
    best_columns: dict[str, str] = emb.get("best_columns", {})

    if not l2_scores:
        logging.warning("%s: no l2_scores in log — skipping", slug)
        return None

    ranked_columns = _build_ranked_columns(l2_scores, best_columns)

    result = await runner.run_llm(
        role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
        prompt_parameters={
            "query": query,
            "ranked_columns": ranked_columns,
        },
        output_type=BoundaryResult,
    )

    prec, rec, f1, tp, fp, fn = _compute_metrics(relevant, result.relevant_agent_names)
    missed = sorted(set(relevant) - set(result.relevant_agent_names))
    extra_selected = sorted(set(result.relevant_agent_names) - set(relevant))

    icon = "✓" if not missed else "✗"
    print(f"  {icon} {slug}")
    print(f"       relevant={relevant}")
    print(f"       llm_selected={result.relevant_agent_names}")
    print(f"       missed={missed}  extra={extra_selected}")
    print(f"       precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"       reasoning: {result.reasoning}")
    print()

    return {
        "file": slug,
        "relevant": relevant,
        "llm_selected": result.relevant_agent_names,
        "missed": missed,
        "extra_selected": extra_selected,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_agents": len(l2_scores),
    }


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    macro_p = sum(r["precision"] for r in results) / len(results)
    macro_r = sum(r["recall"] for r in results) / len(results)
    macro_f = sum(r["f1"] for r in results) / len(results)
    micro_tp = sum(r["tp"] for r in results)
    micro_fp = sum(r["fp"] for r in results)
    micro_fn = sum(r["fn"] for r in results)
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    perfect = sum(1 for r in results if r["recall"] == 1.0)

    avg_relevant = sum(len(r["relevant"]) for r in results) / len(results)
    avg_total = sum(r["total_agents"] for r in results) / len(results)

    print("=" * 60)
    print(f"  Queries evaluated : {len(results)}")
    print(f"  Perfect recall    : {perfect}/{len(results)}")
    print(f"  Avg relevant      : {avg_relevant:.1f} / {avg_total:.0f} total agents")
    print()
    print("  -- LLM boundary finder --")
    print(f"  Macro precision : {macro_p:.3f}  |  Micro precision : {micro_p:.3f}")
    print(f"  Macro recall    : {macro_r:.3f}  |  Micro recall    : {micro_r:.3f}")
    print(f"  Macro F1        : {macro_f:.3f}  |  Micro F1        : {micro_f:.3f}")
    print("=" * 60)


async def _main(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    input_dir = Path(args.input_dir) if args.input_dir else None
    config = load_config(args.config)

    query_dirs = sorted(
        d for d in log_dir.iterdir()
        if d.is_dir() and any(d.glob("*/stage_1_output_embedding_pre_filter.json"))
    )
    if not query_dirs:
        print(f"No embedding pre-filter logs found in {log_dir}")
        sys.exit(1)

    print(f"Evaluating LLM boundary finder over {len(query_dirs)} queries in {log_dir}\n")

    results: list[dict] = []
    with obs_context.pipeline_context(
        name="eval_llm_boundary",
        trace_name="eval_llm_boundary",
        output_dir=None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        data_dir=log_dir,
        cache_mode=config.cache_mode,
        config=config,
        trace_metadata={},
    ):
        for query_dir in query_dirs:
            result = await _process_query(query_dir, input_dir)
            if result is not None:
                results.append(result)

    _print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM boundary finding on ranked column lists")
    parser.add_argument("--log_dir", required=True, help="Log dir with per-query subdirectories")
    parser.add_argument("--input_dir", default=None, help="Input dir with original JSON files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cache_dir", default=None, help="Cache dir for LLM calls")
    parser.add_argument("--output", default=None, help="Path to write JSON results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
