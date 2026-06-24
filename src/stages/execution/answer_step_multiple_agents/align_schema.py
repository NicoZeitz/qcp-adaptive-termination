import uuid
from collections import OrderedDict
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
    function_tool,
)
from pandas import DataFrame


from qtypes.aligned_result_list import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
)
from qtypes.data_agent import DataAgent
from stages.common.agent_configuration import get_coordinator_agent_configuration
from stages.common.data_agent.data_description import data_agent_data_description
from stages.execution.validation import validate_schema_output
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


async def align_schema(
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList:
    agent_descriptions: list[dict[str, str]] = []
    for name in agent_names:
        agent = next(a for a in data_agents if a.name == name)
        description = await data_agent_data_description(agent)
        agent_descriptions.append({"name": name, "description": description})

    agents_description_text = "\n\n".join(
        f"Agent: {a['name']}\n{a['description']}" for a in agent_descriptions
    )

    source_agent_column = f"source_agent_{uuid.uuid4().hex[:8]}"
    schema_metadata: AlignedSchemaMetadata | None = None

    @function_tool
    def set_schema(
        columns: list[str],
        aggregation_type: AggregationType,
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
            aggregation_type,
        )

        schema_metadata = AlignedSchemaMetadata(
            aggregation_type=aggregation_type,
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

    with own_custom_span(
        "Query Coordinator - Align Schema",
        input={
            "query": query,
            "merge_task": merge_task,
            "agent_names": agent_names,
        },
    ):
        schema_metadata = await run_agent(
            agent_configuration=get_coordinator_agent_configuration("Align Schema"),
            prompt_name="execution-coordinator-align-schema",
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

        append_to_span_output({"schema_metadata": schema_metadata})
        return AlignedResultList(
            data=DataFrame(columns=schema_metadata.columns + [source_agent_column]),
            schema_metadata=schema_metadata,
        )
