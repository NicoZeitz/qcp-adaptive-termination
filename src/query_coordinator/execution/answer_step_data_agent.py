from data_agent.data_agent import DataAgent
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan.query_plan import QueryPlanStep


async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    responsible_data_agent = next(
        (agent for agent in data_agents if agent.name == query_plan_step.to_agent)
    )
    return await responsible_data_agent.answer_query(
        query_plan_step.query, additional_information
    )
