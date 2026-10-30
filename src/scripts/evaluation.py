"""CLI entrypoint for evaluation scripts.

Usage:
    python -m scripts.evaluation selection-strategies --dataset_dir ... --output_dir ...
    python -m scripts.evaluation completeness --dataset_dir ... --output_dir ...
"""

import asyncio
import csv
import logging
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType

import yaml

from qcp_adaptive_termination import load_config
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.observability import context
from scripts import completeness_evaluation, k_warmup, selection_strategy_evaluation
from scripts.utils.orchestrator import run_parallel

_SUBCOMMANDS: dict[str, tuple[ModuleType, str]] = {
    "selection-strategies": (
        selection_strategy_evaluation,
        "Benchmark all selection strategies across query steps",
    ),
    "completeness": (completeness_evaluation, "Evaluate the completeness evaluator across query steps"),
    "k-warmup": (k_warmup, "Study where the unknown-unknowns estimator stabilizes per ordering"),
}


def _parse_arguments() -> Namespace:
    """Build the CLI parser with an evaluation subcommand per benchmark."""
    parser = ArgumentParser(prog="qcp-adaptive-termination-evaluation", description="Evaluation scripts for QCP")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name, (_, help_text) in _SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--config",
            type=str,
            default=None,
            help="Config as a YAML file path or inline YAML string",
        )
        sub.add_argument(
            "--config_override",
            type=str,
            action="append",
            default=None,
            help="YAML file path or inline YAML string deep-merged onto --config, repeatable (applied in order)",
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
            help="Dataset folder containing _meta/<input folder of the subcommand> and _meta/meta_mapping.yaml",
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

    completeness = subparsers.choices["completeness"]
    completeness.add_argument(
        "--continue",
        action="store_true",
        help="completeness only: rerun queries that previously failed in --output_dir, "
        "clearing their per-query folder first, then re-aggregate over all queries.",
    )

    return parser.parse_args()


def _read_yaml(source: str) -> dict:
    """Read a YAML mapping from a file path or an inline YAML string."""
    path = Path(source)
    raw = yaml.safe_load(path.read_text(encoding="utf-8") if len(source) <= 255 and path.is_file() else source)
    return raw or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        merged[key] = (
            _deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def _save_config(output_dir: Path, source: str | None, overrides: list[str], config: Config):
    """Copy the input config (and overrides) and write the resolved config into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    named = [("config.yaml", source)] + [(f"config_override_{i}.yaml", o) for i, o in enumerate(overrides, 1)]
    for name, text in named:
        if text is None:
            continue
        if len(text) <= 255 and Path(text).is_file():
            shutil.copy(text, output_dir / name)
        else:
            (output_dir / name).write_text(text, encoding="utf-8")
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)


def _failed_queries(status_csv: Path) -> list[str]:
    """Query slugs whose completeness status is not ``complete``."""
    with status_csv.open(encoding="utf-8") as f:
        return [row["query"] for row in csv.DictReader(f) if row.get("status") != "complete"]


def _clear_outputs(slugs: list[str], output_dir: Path):
    """Remove each failed query's per-query output folder so it starts clean."""
    for slug in slugs:
        shutil.rmtree(output_dir / slug, ignore_errors=True)


async def _run_subcommand(args: Namespace):
    """Load config and run the selected evaluation subcommand."""

    output_dir = Path(args.output_dir)

    with context.logging_context(output_dir):
        config = load_config(args.config)
        overrides: list[str] = args.config_override or []
        if overrides:
            merged = config.model_dump(mode="json")
            for override in overrides:
                merged = _deep_merge(merged, _read_yaml(override))
            config = Config.model_validate(merged)
        _save_config(output_dir, args.config, overrides, config)

        dataset_dir = Path(args.dataset_dir)

        module, _ = _SUBCOMMANDS[args.subcommand]
        input_dir = dataset_dir / "_meta" / module.INPUT_DIR
        input_files = sorted(input_dir.glob("*.json"))
        if not input_files:
            raise FileNotFoundError(f"No input files in {input_dir}")
        if args.query:
            needle = args.query.lower()
            input_files = [f for f in input_files if needle in f.stem.lower()]

        if getattr(args, "continue", False):
            if args.subcommand != "completeness":
                raise SystemExit("--continue is only supported for the completeness subcommand")
            status_csv = output_dir / "evaluator_run_status.csv"
            if not status_csv.exists():
                raise SystemExit(f"--continue requires a previous run with {status_csv} in --output_dir")
            failed = _failed_queries(status_csv)
            _clear_outputs(failed, output_dir)
            if not failed:
                logging.info("Nothing to rerun: no failed queries in %s", output_dir)
                return
            failed_set = set(failed)
            input_files = [f for f in input_files if f.stem in failed_set]
            logging.info("Rerunning %d failed queries into %s", len(input_files), output_dir)

        await run_parallel(
            module.run_query,
            input_files,
            config=config,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            parallel=not args.sequential,
            max_workers=args.max_workers,
        )
        module.aggregate(output_dir, input_dir)


def main():
    """Parse CLI args and run the requested evaluation."""
    args = _parse_arguments()
    asyncio.run(_run_subcommand(args))


if __name__ == "__main__":
    main()
