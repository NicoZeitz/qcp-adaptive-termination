import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from agents import trace
from langfuse import get_client

from qcp_adaptive_termination.cache import load_cache_file, setup_cache
from qcp_adaptive_termination.llm.providers import setup_providers
from qcp_adaptive_termination.models.config import Config, config_context, config_var
from qcp_adaptive_termination.models.stages.collaborative_query_answering import (
    CollaborativeQueryAnswering,
)
from qcp_adaptive_termination.models.stages.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from qcp_adaptive_termination.models.stages.drafted_query_plan import DraftedQueryPlan
from qcp_adaptive_termination.models.stages.filtered_relevant_agents import (
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
from qcp_adaptive_termination.tracing import (
    setup_langfuse,
    setup_logging,
    setup_tracing,
)

STAGES = {
    1: DISCOVERY_STAGE_NAME,
    2: PLANNING_STAGE_NAME,
    3: EXECUTION_STAGE_NAME,
}


async def run_pipeline(
    name: str,
    input_file: Path,
    config: Config | None = None,
    start_stage: int = 1,
    end_stage: int = 3,
) -> CollaborativeQueryAnsweringResult | None:
    cfg = config or Config()

    if cfg.log_dir is None:
        raise ValueError("log_dir must be set in Config")
    if cfg.cache_dir is None:
        raise ValueError("cache_dir must be set in Config")

    log_dir = Path(cfg.log_dir)
    cache_dir = Path(cfg.cache_dir)
    log_name = cfg.log_name or datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    if start_stage > end_stage:
        raise ValueError("start_stage must be <= end_stage")

    log_filename = f"Stage {start_stage}-{end_stage} - {log_name}"

    load_dotenv(override=True)
    setup_logging(log_filename, log_dir)
    setup_cache(cache_dir, cfg.cache_mode)
    setup_providers()
    if not setup_langfuse():
        sys.exit(1)
    setup_tracing(log_filename, log_dir)

    with config_context(cfg):
        return await _run_stages(
            name=name,
            input_file=input_file,
            log_dir=log_dir,
            log_filename=log_filename,
            start_stage=start_stage,
            end_stage=end_stage,
        )


async def run_query(
    name: str,
    input_file: Path,
    config: Config | None = None,
) -> CollaborativeQueryAnsweringResult | None:
    return await run_pipeline(
        name=name,
        input_file=input_file,
        config=config,
        start_stage=1,
        end_stage=3,
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
            start_stage += 1
            if end_stage < start_stage:
                return None

        if start_stage == 2:
            data = cast(FilteredRelevantAgents, data)
            data = await draft_query_plan(
                data.query, data.data_agents, data.relevant_data_agents
            )
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
