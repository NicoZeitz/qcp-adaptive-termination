import asyncio
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from qcp_adaptive_termination.models import result_type
from qcp_adaptive_termination.pipeline import load_config, run_pipeline


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
        "--data_dir",
        type=str,
        default=None,
        help="Root directory for agent data files (default: parent of input file)",
    )
    args = parser.parse_args()

    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    config = load_config(args.config)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) / datetime_str if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else input_file.parent

    result = asyncio.run(
        run_pipeline(
            name=f"{args.name} - {datetime_str}" if args.name else datetime_str,
            input_file=input_file,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            cache_mode=config.cache_mode,
            config=config,
        )
    )

    if args.end_stage == 3:
        print("Pipeline completed successfully. Result preview:")
        if result_type.is_result_type(result):
            print(json.dumps(result_type.serialize(result, context="preview"), indent=2, ensure_ascii=False))
        else:
            print("No result produced.")


if __name__ == "__main__":
    main()
