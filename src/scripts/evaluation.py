"""CLI entrypoint for evaluation scripts.

Usage:
    python -m scripts.evaluation selection-strategies --dataset_dir ... --output_dir ...
    python -m scripts.evaluation completeness --dataset_dir ... --output_dir ...
"""

import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path

from qcp_adaptive_termination import load_config
from qcp_adaptive_termination.observability import context
from scripts import completeness_evaluation, k_warmup, selection_strategy_evaluation
from scripts.k_warmup.input_gen import generate_inputs
from scripts.utils.orchestrator import run_parallel


def _parse_arguments() -> Namespace:
    """Build the CLI parser with an evaluation subcommand per benchmark."""
    parser = ArgumentParser(prog="qcp-adaptive-termination-evaluation", description="Evaluation scripts for QCP")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name, help_text in (
        ("selection-strategies", "Benchmark all selection strategies across query steps"),
        ("completeness", "Evaluate the completeness evaluator across query steps"),
        ("k-warmup", "Study where the unknown-unknowns estimator stabilizes per ordering"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--config",
            type=str,
            default=None,
            help="Config as a YAML file path or inline YAML string",
        )
        sub.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Directory to write per-query + aggregate results",
        )
        sub.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Directory for cache files",
        )
        sub.add_argument(
            "--dataset_dir",
            type=str,
            default=None,
            help="Dataset folder containing _meta/input (query JSONs) and _meta/meta_mapping.yaml",
        )
        sub.add_argument(
            "--generate_inputs",
            action="store_true",
            help="(k-warmup) generate per-table single-step input files into _meta/input_kwarmup",
        )
        sub.add_argument(
            "--workers",
            type=int,
            default=None,
            help="(k-warmup) parallel ordering workers per table",
        )
        sub.add_argument(
            "--query",
            type=str,
            default=None,
            help="Only run input queries whose filename stem contains this substring (case-insensitive).",
        )

        sub.add_argument("--parallel", action="store_true", default=True, help="Run queries in parallel (default)")
        sub.add_argument("--sequential", action="store_true", help="Run queries sequentially")
        sub.add_argument("--max_workers", type=int, default=None, help="Max parallel worker processes")

    return parser.parse_args()


def _functions_for_subcommand(subcommand: str):
    """Return the run_query worker+aggregate pair for an evaluation subcommand."""
    if subcommand == "selection-strategies":
        return selection_strategy_evaluation.run_query, selection_strategy_evaluation.aggregate
    if subcommand == "completeness":
        return completeness_evaluation.run_query, completeness_evaluation.aggregate
    if subcommand == "k-warmup":
        return k_warmup.run_query, k_warmup.aggregate
    raise ValueError(f"Unknown subcommand: {subcommand}")


async def _run_subcommand(args: Namespace):
    """Load config and run the selected evaluation subcommand."""

    output_dir = Path(args.output_dir)

    with context.logging_context(output_dir):
        config = load_config(args.config)

        dataset_dir = Path(args.dataset_dir)

        run_query, aggregate = _functions_for_subcommand(args.subcommand)
        input_dir = dataset_dir / "_meta" / "input"

        if args.subcommand == "k-warmup":
            # For k-warmup, generate per-table single-step inputs (unless already present)
            # into a dedicated input dir, then run against those.
            kwargs_dir = dataset_dir / "_meta" / "input_kwarmup"
            if args.generate_inputs or not list(kwargs_dir.glob("*.json")):
                from qcp_adaptive_termination.pipeline import load_input

                template_path = next(iter(input_dir.glob("*.json")))
                _data, _ = load_input(template_path)
                from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering

                assert isinstance(_data, CollaborativeQueryAnswering)
                generate_inputs(
                    kwargs_dir,
                    dataset_dir,
                    _data.data_agents,
                    workers=args.workers,
                )
            input_dir = kwargs_dir

        input_files = sorted(input_dir.glob("*.json"))
        if args.query:
            needle = args.query.lower()
            input_files = [f for f in input_files if needle in f.stem.lower()]

        await run_parallel(
            run_query,
            input_files,
            config=config,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            parallel=not args.sequential,
            max_workers=args.max_workers,
        )
        aggregate(output_dir, input_dir)


def main():
    """Parse CLI args and run the requested evaluation."""
    args = _parse_arguments()
    asyncio.run(_run_subcommand(args))


if __name__ == "__main__":
    main()
