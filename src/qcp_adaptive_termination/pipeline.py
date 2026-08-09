import logging
import shutil
from pathlib import Path
from typing import Any, Literal, cast, overload

import yaml

import qcp_adaptive_termination.stages as stages
from qcp_adaptive_termination.models import CacheMode, Config
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
)
from qcp_adaptive_termination.observability import cache, context


def load_config(source: str | Path | None) -> Config:
    if source is None:
        return Config()

    config_path = Path(source)
    raw_config = yaml.safe_load(config_path.read_text() if config_path.exists() else str(source))
    return Config.model_validate(raw_config)


def load_input(path: str | Path) -> Any:
    data, extra = cache.load_cache_file(
        Path(path),
        deserializable_classes=[
            CollaborativeQueryAnswering,
            FilteredRelevantAgents,
            DraftedQueryPlan,
            CollaborativeQueryAnsweringResult,
        ],
    )
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

    if output_dir := context.get_output_dir():
        # copy input file to output dir for record-keeping
        output_file = output_dir / input_file.name
        shutil.copy(input_file, output_file)

        # copy the config file to the output dir for record-keeping
        if config := context.get_config():
            config_output_file = output_dir / "config.yaml"
            with open(config_output_file, "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, sort_keys=False, allow_unicode=True, default_flow_style=False)

        # log the config
        logging.info(f"Config: {config.model_dump_json(ensure_ascii=False)}")

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

        # TODO: better code for this (e.g. hooks)
        # logging.info(preview.result_to_json_preview(result.result))

        # if output_dir and log_name:
        #     output_path = output_dir / f"{log_name} - final_result.json"
        #     with open(output_path, "w") as f:
        #         f.write(result.model_dump_json(indent=4, ensure_ascii=False))

        #     from qcp_adaptive_termination.stages.execution.multi_agent import (
        #         MULTI_AGENT_ITERATION_RECORDS,
        #     )

        #     for execution_id, records in MULTI_AGENT_ITERATION_RECORDS.items():
        #         records_path = output_dir / f"{log_name} - iteration_records - {execution_id}.json"
        #         with open(records_path, "w", encoding="utf-8") as f:
        #             json.dump(
        #                 [r.model_dump() for r in records],
        #                 f,
        #                 indent=4,
        #                 ensure_ascii=False,
        #                 default=str,
        #             )

        #         run_prefix = str(output_dir / log_name)
        #         try:
        #             from scripts.plot_iteration_records import (
        #                 extract_series,
        #                 load_final_result_value,
        #                 load_reasoning_by_agents_queried,
        #                 load_records,
        #                 plot,
        #                 save_reasoning_text,
        #             )

        #             plot_records = load_records(str(records_path))
        #             series = extract_series(plot_records)
        #             count_gt = series["last_observed_count"] or 0
        #             value_gt = (
        #                 (extra.get("full_answer") if extra else None)
        #                 or load_final_result_value(run_prefix)
        #                 or series["last_observed_value"]
        #                 or 0
        #             )

        #             traces_file = output_dir / f"{log_name}_traces.jsonl"
        #             reasoning_by_n = None
        #             if traces_file.exists():
        #                 reasoning_by_n = load_reasoning_by_agents_queried(str(traces_file))
        #                 if reasoning_by_n:
        #                     save_reasoning_text(
        #                         series,
        #                         reasoning_by_n,
        #                         run_prefix + " - completeness reasoning.txt",
        #                     )

        #             plot(
        #                 series,
        #                 count_gt,
        #                 value_gt,
        #                 run_prefix,
        #                 include_confidence_intervals=cfg.include_confidence_intervals,
        #             )
        #         except Exception as e:
        #             logging.warning(f"Plotting failed for {execution_id}: {e}")

        # return result
