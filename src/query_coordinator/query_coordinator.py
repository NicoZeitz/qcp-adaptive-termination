from abc import ABC, abstractmethod
import asyncio

from openai.types.shared import Reasoning

from qtypes.agent_configuration import AgentConfiguration
from data_agent.data_agent import DataAgent
from qtypes.query_plan.query_plan import QueryPlan
from qtypes.query_plan.query_plan_step_execution_result import (
    FailedExecution,
    QueryPlanStepExecutionResult,
    SuccessfulExecution,
)
from qtypes.query_plan.result import ResultType, get_failed_result
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


class QueryCoordinator(ABC):
    def __init__(self, name: str = "Query Coordinator"):
        self.name = name

    @property
    def agent_configuration(self) -> AgentConfiguration:
        return AgentConfiguration(
            name=self.name,
            model="gpt-5",
            reasoning=Reasoning(effort="high"),
        )

    async def collaborative_query_answering(
        self, query: str, data_agents: list[DataAgent]
    ) -> ResultType:
        # Stage 1: Filter relevant agents by Query Coordinator
        data_agents = await self.filter_relevant_agents(data_agents, query)

        # Stage 2: Filter relevant agents by asking the agents directly
        data_agents = await self.ask_relevant_agents(data_agents, query)

        # Stage 3: Draft a Query Plan
        query_plan = await self.draft_query_plan(data_agents, query)

        # Stage 4: Execute a Query Plan
        result = await self.execute_query_plan(data_agents, query_plan)

        return result

    async def execute_query_plan(
        self, data_agents: list[DataAgent], query_plan: QueryPlan
    ) -> ResultType:
        MAX_REPLAN_ATTEMPTS = 2
        failed_execution = None

        for replan_attempts in range(MAX_REPLAN_ATTEMPTS + 1):
            with own_custom_span(f"{replan_attempts} - Execute Query Plan") as span:
                append_to_span_input(
                    span,
                    {
                        "query_plan": query_plan.to_span_data(),
                        "data_agents": [agent.name for agent in data_agents],
                    },
                )

                while True:
                    execution_result = await self.execute_query_plan_next_step(
                        data_agents, query_plan
                    )

                    if (
                        isinstance(execution_result, SuccessfulExecution)
                        and execution_result.execution_finished
                    ):
                        return execution_result.result
                    elif isinstance(execution_result, FailedExecution):
                        failed_execution = execution_result
                        break  # Replan if execution failed

            if replan_attempts > MAX_REPLAN_ATTEMPTS:
                return get_failed_result()

            with own_custom_span(f"{replan_attempts} - Replan") as span:
                append_to_span_input(
                    span,
                    {
                        "query_plan": query_plan.to_span_data(),
                        "data_agents": [agent.name for agent in data_agents],
                        "reason": failed_execution,
                    },
                )
                # TODO: re-add agents
                query_plan = await self.replan_query_plan(
                    data_agents=data_agents,
                    query_plan=query_plan,
                    reason=failed_execution,
                )

                append_to_span_output(span, {"query_plan": query_plan.to_span_data()})

        return get_failed_result()

    @abstractmethod
    async def filter_relevant_agents(
        self, data_agents: list[DataAgent], query: str
    ) -> list[DataAgent]:
        pass

    @abstractmethod
    async def ask_relevant_agents(
        self, data_agents: list[DataAgent], query: str
    ) -> list[DataAgent]:
        pass

    @abstractmethod
    async def draft_query_plan(
        self, data_agents: list[DataAgent], query: str
    ) -> QueryPlan:
        pass

    @abstractmethod
    async def replan_query_plan(
        self,
        data_agents: list[DataAgent],
        query_plan: QueryPlan,
        reason: FailedExecution,
    ) -> QueryPlan:
        pass

    @abstractmethod
    async def execute_query_plan_next_step(
        self, data_agents: list[DataAgent], query_plan: QueryPlan
    ) -> QueryPlanStepExecutionResult:
        pass
