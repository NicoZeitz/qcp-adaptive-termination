import argparse
import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast, overload

import src.qcp_adaptive_termination.cli as cli
from qcp_adaptive_termination import run_stages
from qcp_adaptive_termination.models import Config, DataAgent
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
    Stage,
)
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, load_input


@dataclass(slots=True)
class ScriptContext:
    args: argparse.Namespace
    datetime_str: str
    config: Config
    input_file: Path
    output_dir: Path | None
    cache_dir: Path | None
    data_dir: Path
    start_stage: Stage
    end_stage: Stage
    name: str
    trace_name: str
    data: CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult
    extra: dict[str, Any] | None

    @property
    def data_agents(self) -> list[DataAgent]:
        if isinstance(self.data, (CollaborativeQueryAnswering, FilteredRelevantAgents, DraftedQueryPlan)):
            return self.data.data_agents
        raise TypeError(f"Input type {type(self.data).__name__} has no data_agents")


@overload
def run_script[T](
    *,
    run_fn: Callable[[ScriptContext], Coroutine[Any, Any, T]],
    prog: str = "qcp-adaptive-termination",
    description="Collaborative Query Processing with Adaptive Termination",
    before_fn: Callable[[ScriptContext], Coroutine[Any, Any, None]] | None = None,
    after_fn: Callable[[ScriptContext, T], Coroutine[Any, Any, None]] | None = None,
) -> T: ...
@overload
def run_script(
    *,
    run_fn: None = None,
    prog: str = "qcp-adaptive-termination",
    description="Collaborative Query Processing with Adaptive Termination",
    before_fn: Callable[[ScriptContext], Coroutine[Any, Any, None]] | None = None,
    after_fn: Callable[
        [ScriptContext, FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult],
        Coroutine[Any, Any, None],
    ]
    | None = None,
) -> FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult: ...
def run_script(
    *,
    run_fn=None,
    prog: str = "qcp-adaptive-termination",
    description: str = "Collaborative Query Processing with Adaptive Termination",
    before_fn=None,
    after_fn=None,
):
    args = cli.argument_parser(prog, description).parse_args()

    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    config = load_config(args.config)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else input_file.parent.parent.parent
    data_dir = dataset_dir.parent

    start_stage = cast(Stage, args.start_stage)
    end_stage = cast(Stage, args.end_stage)
    name = f"{args.name} - {datetime_str}" if args.name else datetime_str
    cache_mode = config.cache_mode

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    cli.register_default_output_hooks()

    if start_stage.order > end_stage.order:
        raise ValueError("start_stage must be <= end_stage")

    max_trace_label_len = 50
    name_part = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    trace_name = f"[QCP] {name_part}{input_file.stem[:max_trace_label_len]}: Stage {start_stage}-{end_stage}"

    data, extra = load_input(input_file)

    async def _runner():
        with context.pipeline_context(
            name=name,
            trace_name=trace_name,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            cache_mode=cache_mode,
            config=config,
            trace_metadata={
                "start_stage": start_stage.value,
                "end_stage": end_stage.value,
                "input_file": str(input_file),
            },
        ):
            ctx = ScriptContext(
                args=args,
                datetime_str=datetime_str,
                config=config,
                input_file=input_file,
                output_dir=output_dir,
                cache_dir=cache_dir,
                data_dir=data_dir,
                start_stage=start_stage,
                end_stage=end_stage,
                name=name,
                trace_name=trace_name,
                data=data,
                extra=extra,
            )

            if before_fn is not None:
                await before_fn(ctx)

            if run_fn is not None:
                result = await run_fn(ctx)
            else:
                result = await run_stages(
                    input_file=input_file,
                    start_stage=start_stage,
                    end_stage=end_stage,
                )

            if after_fn is not None:
                await after_fn(ctx, result)

            return result

    return asyncio.run(_runner())
