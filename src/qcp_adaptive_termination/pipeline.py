from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast, overload

import yaml

import qcp_adaptive_termination.hooks as hooks
import qcp_adaptive_termination.stages as stages
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import CacheMode, Config
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
)
from qcp_adaptive_termination.observability import cache, context


@lru_cache(maxsize=32)
def load_config(source: str | Path | None) -> Config:
    if source is None:
        return Config()

    config_path = Path(source)
    is_file = len(str(source)) <= 255 and config_path.exists()
    raw_config = yaml.safe_load(config_path.read_text() if is_file else str(source))
    config = Config.model_validate(raw_config)
    hooks.emit(HookEvent.LOAD_CONFIG, config=config)
    return config


@lru_cache(maxsize=32)
def load_input(
    path: str | Path,
) -> tuple[
    CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult,
    dict[str, Any] | None,
]:
    data, extra = cache.load_cache_file(
        Path(path),
        deserializable_classes=[
            CollaborativeQueryAnswering,
            FilteredRelevantAgents,
            DraftedQueryPlan,
            CollaborativeQueryAnsweringResult,
        ],
    )
    hooks.emit(HookEvent.LOAD_INPUT, input_file=Path(path), data=data, extra=extra)
    return data, extra


@overload
async def run_pipeline(
    name: str,
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[1],
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
) -> FilteredRelevantAgents: ...
@overload
async def run_pipeline(
    name: str,
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[2],
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
) -> DraftedQueryPlan: ...
@overload
async def run_pipeline(
    name: str,
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[3],
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
) -> CollaborativeQueryAnsweringResult: ...
async def run_pipeline(
    name: str,
    input_file: Path,
    start_stage: Literal[1, 2, 3] = 1,
    end_stage: Literal[1, 2, 3] = 3,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
) -> FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult:
    if start_stage > end_stage:
        raise ValueError("start_stage must be <= end_stage")

    max_trace_label_len = 50
    name_part = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    trace_name = f"[QCP] {name_part}{input_file.stem[:max_trace_label_len]}: Stage {start_stage}-{end_stage}"

    with context.pipeline_context(
        name=name,
        trace_name=trace_name,
        output_dir=output_dir,
        cache_dir=cache_dir,
        data_dir=data_dir,
        cache_mode=cache_mode,
        config=config,
        trace_metadata={
            "start_stage": start_stage,
            "end_stage": end_stage,
            "input_file": str(input_file),
        },
    ):
        return await run_stages(
            input_file=input_file,
            start_stage=start_stage,
            end_stage=end_stage,
        )


async def run_query(
    name: str,
    input_file: Path,
) -> CollaborativeQueryAnsweringResult:
    return await run_pipeline(
        name=name,
        input_file=input_file,
        start_stage=1,
        end_stage=3,
    )


@overload
async def run_stages(
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[1],
) -> FilteredRelevantAgents: ...
@overload
async def run_stages(
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[2],
) -> DraftedQueryPlan: ...
@overload
async def run_stages(
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[3],
) -> CollaborativeQueryAnsweringResult: ...
async def run_stages(
    input_file: Path,
    start_stage: Literal[1, 2, 3],
    end_stage: Literal[1, 2, 3],
) -> FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult:
    if start_stage > end_stage:
        raise ValueError("start_stage must be <= end_stage")

    data, _extra = load_input(input_file)

    if start_stage == 1:
        data = cast(CollaborativeQueryAnswering, data)
        data = await stages.filter_relevant_agents(data.query, data.data_agents)
        start_stage += 1
        if end_stage < start_stage:
            return data

    if start_stage == 2:
        data = cast(FilteredRelevantAgents, data)
        data = await stages.draft_query_plan(data.query, data.data_agents, data.relevant_data_agents)
        start_stage += 1
        if end_stage < start_stage:
            return data

    if start_stage == 3:
        data = cast(DraftedQueryPlan, data)
        data = await stages.execute_query_plan(
            data.query_plan,
            data.relevant_data_agents,
            data.data_agents,
        )
        return data
