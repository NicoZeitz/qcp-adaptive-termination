"""CLI entrypoint for evaluation scripts.

Usage:
    python -m scripts.evaluation selection-strategies --dataset_dir ... --output_dir ...
    python -m scripts.evaluation estimator --dataset_dir ... --output_dir ...
"""

import argparse
import asyncio
import logging
from pathlib import Path

from qcp_adaptive_termination import load_config
from scripts import completeness_evaluation, selection_strategy_evaluation
from scripts.utils.orchestrator import (
    add_common_args,
    run_parallel,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.evaluation", description="Evaluation scripts")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name, help_text in (
        ("selection-strategies", "Benchmark all selection strategies across query steps"),
        ("estimator", "Evaluate the completeness evaluator across query steps"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        add_common_args(sub)

    selection_sub = subparsers.choices["selection-strategies"]
    selection_sub.add_argument(
        "--strategy_workers",
        type=int,
        default=None,
        help="Parallel strategy worker processes per query step (default: min(cpu, 4)).",
    )

    return parser


async def _run_subcommand(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parallel = not args.sequential
    dataset_dir = Path(args.dataset_dir)
    input_dir = dataset_dir / "_meta" / "input"

    if args.subcommand == "selection-strategies":
        extra = {"strategy_workers": args.strategy_workers} if args.strategy_workers is not None else {}
        await run_parallel(
            selection_strategy_evaluation.run_query,
            dataset_dir=dataset_dir,
            config=config,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            parallel=parallel,
            max_workers=args.max_workers,
            extra=extra,
            query=args.query,
        )
        selection_strategy_evaluation.aggregate(output_dir, input_dir)
    elif args.subcommand == "estimator":
        await run_parallel(
            completeness_evaluation.run_query,
            dataset_dir=dataset_dir,
            config=config,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            parallel=parallel,
            max_workers=args.max_workers,
            query=args.query,
        )
        completeness_evaluation.aggregate(output_dir, input_dir)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _build_parser().parse_args()
    asyncio.run(_run_subcommand(args))


if __name__ == "__main__":
    main()
