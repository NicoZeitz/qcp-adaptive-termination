import asyncio
import json
from argparse import ArgumentParser
from pathlib import Path

from qcp_adaptive_termination.loaders import load_config
from qcp_adaptive_termination.pipeline import run_pipeline
from qcp_adaptive_termination.preview import result_to_json_preview


def main() -> None:
    parser = ArgumentParser(
        prog="qcp-adaptive-termination",
        description="Collaborative Query Processing with Adaptive Termination",
    )
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Stage to start the pipeline from (1, 2, or 3)",
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Stage after which to end the pipeline at (1, 2, or 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config as a YAML file path or inline YAML string",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = asyncio.run(
        run_pipeline(
            name=args.name,
            input_file=Path(args.input),
            config=config,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
        )
    )

    print("Pipeline completed successfully. Result preview:")
    if result is not None:
        print(json.dumps(result_to_json_preview(result.result), indent=2, ensure_ascii=False))
    else:
        print("No result produced.")


if __name__ == "__main__":
    main()
