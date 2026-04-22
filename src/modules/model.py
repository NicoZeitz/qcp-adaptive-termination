from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal
from pydantic import BaseModel, create_model
from pydantic import BaseModel, Field


@dataclass
class AgentContext:
    pass


class PartialQueryAnswer(BaseModel):
    answer: str = Field(
        description="The answer to the user's question based on the database query results if it exists. Can also only contain a partial result. Does not contain information how to query the answer, instead contains the actual answer to the question."
    )
    needs: list[str] | None = Field(
        description="If the question cannot be answered with the current information, return a list of strings describing what information is needed to answer the question. This can include information that is not in the database, but that would be needed to formulate the correct SQL query or to interpret the results."
    )


class DataAgent(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent"""
        pass

    @abstractmethod
    async def get_data_and_query_capabilities(self) -> str:
        """Return a string that describes the agent's data and query capabilities in Natural Language"""
        pass

    @abstractmethod
    async def get_schema_for_query(self, query: str) -> str | None:
        """Return a string that describes the agent's data and query capabilities, such as the database schema."""
        pass

    @abstractmethod
    async def answer_query(
        self, query: str, additional_information: str
    ) -> PartialQueryAnswer:
        """Answer the given query and return the answer as a string. If the query cannot be answered with the current information, return a string needed"""
        pass


class Need(BaseModel):
    need: str
    answer: str


class QueryPlan(BaseModel):
    need: str | None
    question: str
    from_agent: Literal["User", "Coordinator"] | str
    to_agent: Literal["Coordinator"] | str
    needs: list["QueryPlan"] | None
    answered_needs: list[Need] | None

    @staticmethod
    def create_query_plan_model(participating_agents: list[tuple[DataAgent, str]]):
        participating_agents_names = [agent.name() for agent, _ in participating_agents]

        QueryPlanModel = create_model(
            "QueryPlanModel",
            need=(
                str | None,
                Field(
                    ...,
                    description="The specific information need that this question is addressing. This is used to connect the nodes to their parent's question in the query plan. Empty for the root node.",
                ),
            ),
            question=(
                str,
                Field(
                    ...,
                    description="The specific question that is answered by this node.",
                ),
            ),
            from_agent=(
                Literal["User", "Coordinator"],
                Field(
                    ...,
                    description="The agent that asked the question. This can be the User if it's the root node, or the Coordinator if it's a sub-question that needs to be answered in order to answer the parent question.",
                ),
            ),
            to_agent=(
                Literal["Coordinator", *participating_agents_names],
                Field(..., description="The agent to which the question is directed."),
            ),
            needs=(
                List["QueryPlanModel"] | None,
                Field(
                    ...,
                    description="The sub-questions that need to be answered in order to answer the parent question. Should be empty if the question can be answered directly by an agent. Multiple needs are independent from each other. If there is a dependence between the needs, they should be nested accordingly to reflect the dependence relationship. ",
                ),
            ),
        )
        QueryPlanModel.model_rebuild()
        return QueryPlanModel

    @staticmethod
    def map_query_plan_model_to_query_plan(query_plan_model: Any) -> "QueryPlan":
        return QueryPlan(
            need=query_plan_model.need,
            question=query_plan_model.question,
            from_agent=query_plan_model.from_agent,
            to_agent=query_plan_model.to_agent,
            needs=[
                QueryPlan.map_query_plan_model_to_query_plan(need)
                for need in query_plan_model.needs
            ]
            if query_plan_model.needs
            else None,
            answered_needs=None,
        )

    @staticmethod
    def create_query_replan_model(participating_agents: list[tuple[DataAgent, str]]):
        participating_agents_names = [agent.name() for agent, _ in participating_agents]

        QueryReplanModel = create_model(
            "QueryReplanModel",
            need=(
                str | None,
                Field(
                    ...,
                    description="The specific information need that this question is addressing. This is used to connect the nodes to their parent's question in the query plan. Empty for the root node.",
                ),
            ),
            question=(
                str,
                Field(
                    ...,
                    description="The specific question that is answered by this node.",
                ),
            ),
            from_agent=(
                Literal["User", "Coordinator"],
                Field(
                    ...,
                    description="The agent that asked the question. This can be the User if it's the root node, or the Coordinator if it's a sub-question that needs to be answered in order to answer the parent question.",
                ),
            ),
            to_agent=(
                Literal["Coordinator", *participating_agents_names],
                Field(..., description="The agent to which the question is directed."),
            ),
            needs=(
                List["QueryReplanModel"] | None,
                Field(
                    ...,
                    description="The sub-questions that need to be answered in order to answer the parent question. Should be empty if the question can be answered directly by an agent. Multiple needs are independent from each other. If there is a dependence between the needs, they should be nested accordingly to reflect the dependence relationship. ",
                ),
            ),
            answered_needs=(
                List[Need] | None,
                Field(
                    ...,
                    description="The needs that have already been answered in the old query plan. This is used to keep track of which needs have already been answered and to provide the answers to the replanning agent so that it can make informed decisions about how to replan the query plan.",
                ),
            ),
        )
        QueryReplanModel.model_rebuild()
        return QueryReplanModel

    @staticmethod
    def map_query_replan_model_to_query_plan(query_replan_model: Any) -> "QueryPlan":
        return QueryPlan(
            need=query_replan_model.need,
            question=query_replan_model.question,
            from_agent=query_replan_model.from_agent,
            to_agent=query_replan_model.to_agent,
            needs=[
                QueryPlan.map_query_plan_model_to_query_plan(need)
                for need in query_replan_model.needs
            ]
            if query_replan_model.needs
            else None,
            answered_needs=query_replan_model.answered_needs,
        )


class ReplanningNeeded(BaseModel):
    query_node: QueryPlan


class ResponsibleAgentNotFound(ReplanningNeeded):
    pass


class NeedNotMet(ReplanningNeeded):
    partial_answer: PartialQueryAnswer


@dataclass
class ReplanningNeededException(Exception):
    data: ReplanningNeeded
