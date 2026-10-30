"""CLI entry point for qcp-adaptive-termination."""

import asyncio
import json
from argparse import ArgumentParser, Namespace

from qcp_adaptive_termination.models import result_type
from qcp_adaptive_termination.models.stages import DraftedQueryPlan, FilteredRelevantAgents, Stage, SynthesizedResult
from qcp_adaptive_termination.pipeline import CollaborativeQueryAnsweringResult
from scripts.utils.setup import run_script


def _parse_arguments() -> Namespace:
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
        default=Stage.SYNTHESIS,
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
    parser.add_argument(
        "--langfuse",
        action="store_true",
        default=False,
        help="Enable Langfuse tracing (requires LANGFUSE_* env vars)",
    )
    return parser.parse_args()


def _print_result(
    result: FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult | SynthesizedResult,
):
    """Print the pipeline result to stdout."""
    if isinstance(result, FilteredRelevantAgents):
        print(f"Relevant data agents: {result.relevant_data_agents}")
    elif isinstance(result, DraftedQueryPlan):
        print(json.dumps(result.query_plan, indent=2))
    elif isinstance(result, CollaborativeQueryAnsweringResult):
        print(result_type.serialize(result.result, context="full_preview"))
    elif isinstance(result, SynthesizedResult):
        print(result.summary)


def _main():
    """Parse CLI args and run the QCP pipeline."""
    args = _parse_arguments()
    result = asyncio.run(run_script(args=args))
    _print_result(result)


if __name__ == "__main__":
    _main()
