import hashlib
from collections import OrderedDict

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
    function_tool,
)

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    AgentRole,
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
    DataAgent,
)
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.execution.validation import validate_schema_output


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA)
async def align_schema(
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList:
    agents = [a for a in data_agents if a.name in agent_names]
    agents_description_text = await agent_registry.build_registry_string(
        agents, AgentRegistryType.TECHNICAL, include_capabilities=False
    )

    source_agent_column = f"source_agent_{hashlib.sha256(query.encode()).hexdigest()[:8]}"
    schema_metadata: AlignedSchemaMetadata | None = None

    @function_tool
    def set_schema(
        columns: list[str],
        aggregation_types: list[AggregationType],
        keys: list[str],
        value_column: str | None = None,
    ) -> AlignedSchemaMetadata:
        """Define the unified output schema all agents must conform to."""
        nonlocal schema_metadata

        keys_set = OrderedDict.fromkeys(keys)
        columns_set = OrderedDict.fromkeys(columns)

        validate_schema_output(
            set(columns_set),
            set(keys_set),
            value_column,
            source_agent_column,
            aggregation_types,
        )

        schema_metadata = AlignedSchemaMetadata(
            aggregation_types=aggregation_types,
            keys=list(keys_set),
            value_column=value_column,
            columns=list(columns_set),
            source_agent_column=source_agent_column,
        )
        return schema_metadata

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        return ToolsToFinalOutputResult(
            is_final_output=schema_metadata is not None,
            final_output=schema_metadata,
        )

    with tracing.own_span(
        "Query Coordinator - Align Schema",
        input={
            "query": query,
            "merge_task": merge_task,
            "agent_names": agent_names,
        },
    ):
        schema_metadata = await runner.run_agent(
            role=AgentRole.COORDINATOR_ALIGN_SCHEMA,
            prompt_parameters={
                "query": query,
                "merge_task": merge_task,
                "agents_description": agents_description_text,
            },
            tools=[set_schema],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=AlignedSchemaMetadata,
        )

        tracing.append_to_span_output({"schema_metadata": schema_metadata})
        return AlignedResultList(
            data=pd.DataFrame(columns=schema_metadata.columns + [schema_metadata.source_agent_column]),
            schema_metadata=schema_metadata,
        )
