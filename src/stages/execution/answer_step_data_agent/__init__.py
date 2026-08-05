from model.execution.additional_information import AdditionalInformation
from model.estimation.aligned_result_list import AlignedSchemaMetadata
from model.agent.data_agent import DataAgent
from model.execution.partial_query_answer import PartialQueryAnswer
from stages.execution.answer_step_data_agent.answer_query_data_agent import (
    answer_query_data_agent,
)


async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None = None,
) -> PartialQueryAnswer:
    responsible_data_agent = next(
        (agent for agent in data_agents if agent.name == to_agent)
    )
    return await answer_query_data_agent(
        responsible_data_agent, query, additional_information, schema_metadata
    )
