from data_agent.data_agent import DataAgent
from query_coordinator.planning.draft_query_plan import draft_query_plan
from query_coordinator.execution.execute_query_plan_next_step import (
    execute_query_plan_next_step,
)
from query_coordinator.query_coordinator import QueryCoordinator
from query_coordinator.planning.replan_query_plan import replan_query_plan
from qtypes.query_plan.query_plan import QueryPlan
from query_coordinator.filter_relevant_agents import filter_relevant_agents
from query_coordinator.ask_relevant_agents import ask_relevant_agents
from qtypes.query_plan.query_plan_step_execution_result import (
    FailedExecution,
    QueryPlanStepExecutionResult,
)


class DefaultQueryCoordinator(QueryCoordinator):
    def __init__(self, name: str = "Default Query Coordinator"):
        super().__init__(name=name)

    async def filter_relevant_agents(
        self, data_agents: list[DataAgent], query: str
    ) -> list[DataAgent]:
        return await filter_relevant_agents(self, data_agents, query)

    async def ask_relevant_agents(
        self, data_agents: list[DataAgent], query: str
    ) -> list[DataAgent]:
        return await ask_relevant_agents(data_agents, query)

    async def draft_query_plan(
        self, data_agents: list[DataAgent], query: str
    ) -> QueryPlan:
        return await draft_query_plan(self, data_agents, query)

    async def replan_query_plan(
        self,
        data_agents: list[DataAgent],
        query_plan: QueryPlan,
        reason: FailedExecution,
    ) -> QueryPlan:
        return await replan_query_plan(self, data_agents, query_plan, reason)

    async def execute_query_plan_next_step(
        self, data_agents: list[DataAgent], query_plan: QueryPlan
    ) -> QueryPlanStepExecutionResult:
        return await execute_query_plan_next_step(self, data_agents, query_plan)
