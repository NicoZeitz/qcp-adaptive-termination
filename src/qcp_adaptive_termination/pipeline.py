import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from qcp_adaptive_termination.cache import load_cache_file
from qcp_adaptive_termination.initialization import initialize
from qcp_adaptive_termination.model.config import Config, config_var, config_context
from qcp_adaptive_termination.model.stages.collaborative_query_answering import (
    CollaborativeQueryAnswering,
)
from qcp_adaptive_termination.model.stages.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from qcp_adaptive_termination.model.stages.drafted_query_plan import DraftedQueryPlan
from qcp_adaptive_termination.model.stages.filtered_relevant_agents import (
    FilteredRelevantAgents,
)
from qcp_adaptive_termination.preview import result_to_json_preview
from qcp_adaptive_termination.stages.discovery import (
    STAGE_NAME as DISCOVERY_STAGE_NAME,
    filter_relevant_agents,
)
from qcp_adaptive_termination.stages.execution import (
    STAGE_NAME as EXECUTION_STAGE_NAME,
    execute_query_plan,
)
from qcp_adaptive_termination.stages.planning import (
    STAGE_NAME as PLANNING_STAGE_NAME,
    draft_query_plan,
)

STAGES = {
    1: DISCOVERY_STAGE_NAME,
    2: PLANNING_STAGE_NAME,
    3: EXECUTION_STAGE_NAME,
    99: "DEBUG",
}


async def run_pipeline(
    name: str,
    input_file: Path,
    config: Config | None = None,
    start_stage: int = 1,
    end_stage: int = 3,
    log_dir: Path | None = None,
    cache_dir: Path | None = None,
    log_name: str | None = None,
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
) -> CollaborativeQueryAnsweringResult | None:
    cfg = config or Config()
    resolved_log_dir = log_dir or (Path(cfg.log_dir) if cfg.log_dir else None)
    resolved_cache_dir = cache_dir or (Path(cfg.cache_dir) if cfg.cache_dir else None)
    resolved_log_name = (
        log_name or cfg.log_name or datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    )

    if start_stage > end_stage:
        raise ValueError("start_stage must be <= end_stage")

    if resolved_log_dir is None:
        raise ValueError("log_dir must be provided via argument or Config")
    if resolved_cache_dir is None:
        raise ValueError("cache_dir must be provided via argument or Config")

    log_filename = f"Stage {start_stage}-{end_stage} - {resolved_log_name}"

    initialize(
        log_filename,
        resolved_log_dir,
        resolved_cache_dir,
        throw_on_cache_miss,
        disable_cache,
    )

    with config_context(cfg):
        return await _run_stages(
            name=name,
            input_file=input_file,
            log_dir=resolved_log_dir,
            log_filename=log_filename,
            start_stage=start_stage,
            end_stage=end_stage,
        )


async def run_query(
    name: str,
    input_file: Path,
    config: Config | None = None,
    log_dir: Path | None = None,
    cache_dir: Path | None = None,
    log_name: str | None = None,
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
) -> CollaborativeQueryAnsweringResult | None:
    return await run_pipeline(
        name=name,
        input_file=input_file,
        config=config,
        start_stage=1,
        end_stage=3,
        log_dir=log_dir,
        cache_dir=cache_dir,
        log_name=log_name,
        throw_on_cache_miss=throw_on_cache_miss,
        disable_cache=disable_cache,
    )


async def _run_stages(
    name: str,
    input_file: Path,
    log_dir: Path,
    log_filename: str,
    start_stage: int,
    end_stage: int,
) -> CollaborativeQueryAnsweringResult | None:
    max_trace_label_len = 50
    name_prefix = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    input_file_prefix = input_file.stem[:max_trace_label_len]
    trace_label = (
        f"[NZ] {name_prefix}{input_file_prefix}: Stage {start_stage}-{end_stage}"
    )
    cfg = config_var.get()

    logging.info(f"Config: {cfg.model_dump_json()}")
    config_output_path = log_dir / f"{log_filename} - execution_config.json"
    with open(config_output_path, "w") as f:
        f.write(cfg.model_dump_json(indent=4))

    from agents import trace
    from langfuse import get_client

    trace_metadata = {
        "start_stage": str(start_stage),
        "end_stage": str(end_stage),
        "name": name,
        "input_file": str(input_file),
        "execution_config": cfg.model_dump_json(),
    }
    langfuse_input = {**trace_metadata, "execution_config": cfg.model_dump()}
    langfuse = get_client()

    with (
        langfuse.start_as_current_observation(
            name=trace_label, input=langfuse_input, metadata=trace_metadata
        ),
        trace(trace_label, metadata=trace_metadata),
    ):
        data, extra = load_cache_file(
            input_file,
            deserializable_classes=[
                CollaborativeQueryAnswering,
                FilteredRelevantAgents,
                DraftedQueryPlan,
                CollaborativeQueryAnsweringResult,
            ],
        )

        if start_stage == 1:
            data = cast(CollaborativeQueryAnswering, data)
            data = await filter_relevant_agents(data.query, data.data_agents)
            data = cast(FilteredRelevantAgents, data)
            start_stage += 1
            if end_stage < start_stage:
                return None

        if start_stage == 2:
            data = cast(FilteredRelevantAgents, data)
            data = await draft_query_plan(
                data.query, data.data_agents, data.relevant_data_agents
            )
            data = cast(DraftedQueryPlan, data)
            start_stage += 1
            if end_stage < start_stage:
                return None

        if start_stage == 3:
            data = cast(DraftedQueryPlan, data)
            result = await execute_query_plan(
                data.query_plan,
                data.relevant_data_agents,
                data.data_agents,
            )
            result = cast(CollaborativeQueryAnsweringResult, result)

            logging.info(result_to_json_preview(result.result))
            output_path = log_dir / f"{log_filename} - final_result.json"
            with open(output_path, "w") as f:
                f.write(result.model_dump_json(indent=4, ensure_ascii=False))

            from qcp_adaptive_termination.stages.execution.multi_agent import (
                MULTI_AGENT_ITERATION_RECORDS,
            )

            for execution_id, records in MULTI_AGENT_ITERATION_RECORDS.items():
                records_path = (
                    log_dir
                    / f"{log_filename} - iteration_records - {execution_id}.json"
                )
                with open(records_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [r.model_dump() for r in records],
                        f,
                        indent=4,
                        ensure_ascii=False,
                        default=str,
                    )

                run_prefix = str(log_dir / log_filename)
                try:
                    from scripts.plot_iteration_records import (
                        extract_series,
                        load_final_result_value,
                        load_reasoning_by_agents_queried,
                        load_records,
                        plot,
                        save_reasoning_text,
                    )

                    plot_records = load_records(str(records_path))
                    series = extract_series(plot_records)
                    count_gt = series["last_observed_count"] or 0
                    value_gt = (
                        (extra.get("full_answer") if extra else None)
                        or load_final_result_value(run_prefix)
                        or series["last_observed_value"]
                        or 0
                    )

                    traces_file = log_dir / f"{log_filename}_traces.jsonl"
                    reasoning_by_n = None
                    if traces_file.exists():
                        reasoning_by_n = load_reasoning_by_agents_queried(
                            str(traces_file)
                        )
                        if reasoning_by_n:
                            save_reasoning_text(
                                series,
                                reasoning_by_n,
                                run_prefix + " - completeness reasoning.txt",
                            )

                    plot(
                        series,
                        count_gt,
                        value_gt,
                        run_prefix,
                        include_confidence_intervals=cfg.include_confidence_intervals,
                    )
                except Exception as e:
                    logging.warning(f"Plotting failed for {execution_id}: {e}")

            return result

    return None
