"""CLI entry point for qcp-adaptive-termination (script wrapper)."""

import asyncio
import json
from argparse import ArgumentParser
from typing import cast

from qcp_adaptive_termination.models import ResultType, result_type
from qcp_adaptive_termination.models.stages import Stage, SynthesizedResult
from scripts.utils.setup import register_stage_output_hooks, run_script


def argument_parser(
    prog: str = "qcp-adaptive-termination", description="Collaborative Query Processing with Adaptive Termination"
) -> ArgumentParser:
    """Build the CLI argument parser for a QCP run."""
    parser = ArgumentParser(
        prog="qcp-adaptive-termination",
        description="Collaborative Query Processing with Adaptive Termination",
    )
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument(
        "--start-stage",
        type=Stage,
        choices=list(Stage),
        default=Stage.DISCOVERY,
        help="Stage to start the pipeline from",
    )
    parser.add_argument(
        "--end-stage",
        type=Stage,
        choices=list(Stage),
        default=Stage.EXECUTION,
        help="Stage after which to end the pipeline at",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config as a YAML file path or inline YAML string",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for logs and traces (default: .logs/ at project root)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for cache files (default: .cache/ at project root)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset folder containing _meta/input (default: derived from input file)",
    )
    return parser


def main() -> None:
    """Parse CLI args and run the QCP pipeline."""
    args = argument_parser().parse_args()
    register_stage_output_hooks()

    result = asyncio.run(run_script(args=args))
    _print_result(result)


def _print_result(result) -> None:
    """Print the pipeline result to stdout."""
    if result_type.is_result_type(result):
        print("Pipeline completed. Result preview:")
        print(
            json.dumps(
                result_type.serialize(cast(ResultType, result), context="preview"),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif isinstance(result, SynthesizedResult):
        print("Pipeline completed. Result summary:")
        print(result.summary)
        print("Result preview:")
        print(
            json.dumps(
                result_type.serialize(result.result, context="preview"),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
