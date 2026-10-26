"""Top-level pipeline entry points: config/input loading and stage orchestration."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

import qcp_adaptive_termination.hooks as hooks
import qcp_adaptive_termination.stages as stages
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import CacheMode, Config
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
    Stage,
    SynthesizedResult,
)
from qcp_adaptive_termination.observability import cache, context

_InputType = CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult
_OutputType = FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult | SynthesizedResult
_INPUT_TYPES: dict[str, type[_InputType]] = {
    "CollaborativeQueryAnswering": CollaborativeQueryAnswering,
    "FilteredRelevantAgents": FilteredRelevantAgents,
    "DraftedQueryPlan": DraftedQueryPlan,
    "CollaborativeQueryAnsweringResult": CollaborativeQueryAnsweringResult,
}


def _strip_type_tags(value: Any) -> Any:
    """Recursively rebuild a value with internal __type__ keys removed."""
    if isinstance(value, dict):
        return {k: _strip_type_tags(v) for k, v in value.items() if k != "__type__"}
    if isinstance(value, list):
        return [_strip_type_tags(item) for item in value]
    return value


@hooks.hook(HookEvent.SETUP_LOAD_CONFIG)
@lru_cache(maxsize=32)
def load_config(source: str | Path | None) -> Config:
    """Load and parse the config from a file path or a YAML string."""
    if source is None:
        return Config()

    config_path = Path(source)
    is_file = len(str(source)) <= 255 and config_path.exists()
    raw_config = yaml.safe_load(config_path.read_text() if is_file else str(source))
    return Config.model_validate(raw_config)


@hooks.hook(HookEvent.SETUP_LOAD_INPUT)
@lru_cache(maxsize=32)
def load_input(
    path: str | Path,
) -> tuple[
    CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult,
    dict[str, Any] | None,
]:
    """Load a pipeline input file, returning its value and optional extra metadata."""
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        value_data = raw["value"]
        type_name = value_data.get("__type__")
        cls = _INPUT_TYPES.get(type_name)
        if cls is None:
            raise ValueError(f"unknown input type: {type_name!r}")
        data = cls.model_validate(_strip_type_tags(value_data))
        extra = raw.get("extra")
    else:
        data, extra = cache.load_cache_file(path)
    return data, extra


async def run_pipeline(
    name: str,
    input_file: Path,
    start_stage: Stage = Stage.DISCOVERY,
    end_stage: Stage = Stage.EXECUTION,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
) -> _OutputType:
    """Run the pipeline stages from start to end over an input file."""
    if start_stage.order > end_stage.order:
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
            "start_stage": start_stage.value,
            "end_stage": end_stage.value,
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
) -> SynthesizedResult:
    """Run the full discovery-to-synthesis pipeline and return the synthesized result."""
    result = await run_pipeline(
        name=name,
        input_file=input_file,
        start_stage=Stage.DISCOVERY,
        end_stage=Stage.SYNTHESIS,
    )
    return cast(SynthesizedResult, result)


async def run_stages(
    input_file: Path,
    start_stage: Stage = Stage.DISCOVERY,
    end_stage: Stage = Stage.EXECUTION,
) -> _OutputType:
    """Run the selected pipeline stages over an already-loaded input file."""
    if start_stage.order > end_stage.order:
        raise ValueError("start_stage must be <= end_stage")

    data, _extra = load_input(input_file)

    query: str = ""
    query_plan: QueryPlan | None = None

    if start_stage.order <= Stage.DISCOVERY.order <= end_stage.order:
        data = cast(CollaborativeQueryAnswering, data)
        query = data.query
        data = await stages.discovery_stage(data.query, data.data_agents)

    if start_stage.order <= Stage.PLANNING.order <= end_stage.order:
        data = cast(FilteredRelevantAgents, data)
        query = data.query
        data = await stages.planning_stage(data.query, data.data_agents, data.relevant_data_agents)

    if start_stage.order <= Stage.EXECUTION.order <= end_stage.order:
        drafted = cast(DraftedQueryPlan, data)
        query = drafted.query_plan.query
        query_plan = drafted.query_plan
        data = await stages.execute_query_plan(
            query_plan,
            drafted.relevant_data_agents,
            drafted.data_agents,
            planning_continuation=drafted.planning_continuation,
        )

    if start_stage.order <= Stage.SYNTHESIS.order <= end_stage.order:
        # When starting at EXECUTION or SYNTHESIS, derive query/query_plan from the input.
        if query_plan is None and isinstance(data, DraftedQueryPlan):
            query = data.query_plan.query
            query_plan = data.query_plan
        data = cast(CollaborativeQueryAnsweringResult, data)
        data = await stages.synthesize_result(
            query=query,
            query_plan=query_plan,
            result=data.result,
            step_uncertainties=data.step_uncertainties,
        )

    return cast(_OutputType, data)
