from qtypes.additional_information import AdditionalInformation
from qtypes.data_agent import DataAgent
from qtypes.partial_query_answer import PartialQueryAnswer
from stages.execution.answer_step_data_agent.answer_query_sql_data_agent import (
    answer_query_sql_data_agent,
)


async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    responsible_data_agent = next(
        (agent for agent in data_agents if agent.name == to_agent)
    )

    if responsible_data_agent == "sql":
        return await answer_query_sql_data_agent(
            responsible_data_agent, query, additional_information
        )
    else:
        raise ValueError(f"Unsupported data agent type: {responsible_data_agent.type}")
