import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from qcp_adaptive_termination.model.execution_config import ExecutionConfig
from qcp_adaptive_termination.pipeline import run_pipeline


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="qcp-adaptive-termination",
        description="Collaborative Query Processing with Adaptive Termination",
    )
    parser.add_argument("--name", type=str, default="")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/imdb/queries/Who played Rocket in the movie called 'Guardians of the Galaxy'?.json",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3, 99],
        default=1,
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        choices=[1, 2, 3, 99],
        default=3,
    )
    parser.add_argument("--throw-on-cache-miss", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Execution config as a YAML file path or inline YAML string",
    )
    args = parser.parse_args()

    config: ExecutionConfig | None = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text())
        else:
            raw = yaml.safe_load(args.config)
        config = ExecutionConfig.model_validate(raw)

    result = asyncio.run(
        run_pipeline(
            name=args.name,
            input_file=Path(args.input),
            config=config,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            log_dir=Path(args.log_dir) if args.log_dir else None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            throw_on_cache_miss=args.throw_on_cache_miss,
            disable_cache=args.disable_cache,
        )
    )

    if result is not None:
        import json
        from qcp_adaptive_termination.preview import result_to_json_preview

        print(
            json.dumps(
                result_to_json_preview(result.result), indent=2, ensure_ascii=False
            )
        )

    sys.exit(0)


if __name__ == "__main__":
    cli()
