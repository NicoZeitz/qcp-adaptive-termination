from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import json
from types import ModuleType
from typing import Any, Literal
from agents import Agent, CustomSpanData, Span, custom_span
from pydantic import BaseModel, JsonValue, create_model
from pydantic import Field
import uuid

from modules.initialization import langfuse

CHAR_LIMIT_LLM = 600
CHAR_LIMIT_LONG_LLM = 6000
CHAR_LIMIT_SPAN = 50

span_by_id: dict[str | None, int] = {}


def append_to_span_input(span: Span, data: dict[str, Any]):
    span.span_data.data["input"] = {
        **span.span_data.data.get("input", {}),
        **data
    }

def append_to_span_output(span: Span, data: dict[str, Any]):
    span.span_data.data["output"] = {
        **span.span_data.data.get("output", {}),
        **data
    }

@contextmanager
def own_custom_span(span_name: str, data: dict[str, Any] | None = None):
    with custom_span(span_name, data=data) as span:
        parent_depth = span_by_id.get(span.parent_id, 0)
        span_by_id[span.span_id] = parent_depth + 1
        print("  " * parent_depth + span_name)
        yield span

        if "input" in span.span_data.data:
            langfuse.update_current_span(input=span.span_data.data["input"])
        if "output" in span.span_data.data:
            langfuse.update_current_span(output=span.span_data.data["output"])

        metadata = span.span_data.data.copy()
        metadata.pop("input", None)
        metadata.pop("output", None)
        langfuse.update_current_span(metadata=metadata)

def preview_text(text: str, char_limit=600):
    if len(text) <= char_limit:
        return text

    char_limit_start = char_limit // 2
    char_limit_end = char_limit - char_limit_start

    omitted_count = len(text) - (char_limit_start + char_limit_end)

    return (
        text[:char_limit_start]
        + f"\n...[{omitted_count} chars]...\n"
        + text[-char_limit_end:]
    )


class Symbol:
    pass


@dataclass
class AgentContext:
    pass


class Need(BaseModel):
    need: str
    answer: JsonValue

    def to_span_data(self) -> JsonValue:
        return {
            "need": self.need,
            "answer": preview_text(str(self.answer), CHAR_LIMIT_SPAN),
        }


class PartialQueryAnswer(BaseModel):
    answer: JsonValue = Field(
        description="The answer to the user's question based on the database query results if it exists. Can also only contain a partial result. Does not contain information how to query the answer, instead contains the actual answer to the question."
    )
    needs: list[JsonValue] | None = Field(
        description="If the question cannot be answered with the current information, return a list of strings describing what information is needed to answer the question. This can include information that is not in the database, but that would be needed to formulate the correct SQL query or to interpret the results."
    )

    def to_span_data(self) -> JsonValue:
        return {
            "answer": preview_text(str(self.answer), CHAR_LIMIT_SPAN),
            "needs": [
                preview_text(str(need), CHAR_LIMIT_SPAN) for need in self.needs or []
            ],
        }


class DataAgent(ABC):
    def __init__(self, agent: Agent[AgentContext]):
        self.agent = agent

    def name(self) -> str:
        """Return the name of the agent"""
        return self.agent.name

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
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        """Answer the given query and return the answer as a string. If the query cannot be answered with the current information, return a string needed"""
        pass


class QueryPlan(BaseModel):
    id: str
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
                list["QueryPlanModel"] | None,
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
            id=str(uuid.uuid4()),
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

    def to_replan_context(self) -> dict:
        """Compact representation for replan context: answered_needs shown as previews."""

        def make_preview(need: "Need") -> dict:
            s = json.dumps(need.answer, default=str)
            return {
                "need": need.need,
                "answer_preview": preview_text(s, CHAR_LIMIT_LLM),
            }

        return {
            "id": self.id,
            "need": self.need,
            "question": self.question,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "needs": [n.to_replan_context() for n in self.needs]
            if self.needs
            else None,
            "answered_needs": [make_preview(n) for n in self.answered_needs]
            if self.answered_needs
            else None,
        }

    @staticmethod
    def collect_answered_needs(plan: "QueryPlan") -> dict[str, list["Need"]]:
        """Build a flat {id: answered_needs} index from a query plan tree."""
        index: dict[str, list[Need]] = {}
        if plan.answered_needs:
            index[plan.id] = plan.answered_needs
        if plan.needs:
            for child in plan.needs:
                index.update(QueryPlan.collect_answered_needs(child))
        return index

    @staticmethod
    def restore_answered_needs(
        plan: "QueryPlan", id_index: dict[str, list["Need"]]
    ) -> None:
        """Restore full answered_needs for nodes whose IDs appear in id_index."""
        if plan.id in id_index:
            plan.answered_needs = id_index[plan.id]
        if plan.needs:
            for child in plan.needs:
                QueryPlan.restore_answered_needs(child, id_index)

    @staticmethod
    def create_query_replan_model(participating_agents: list[tuple[DataAgent, str]]):
        participating_agents_names = [agent.name() for agent, _ in participating_agents]

        QueryReplanModel = create_model(
            "QueryReplanModel",
            id=(
                str | None,
                Field(
                    ...,
                    description="A unique id referencing a existing node in the old query plan. Left empty if it is a newly created node",
                ),
            ),
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
                list["QueryReplanModel"] | None,
                Field(
                    ...,
                    description="The sub-questions that need to be answered in order to answer the parent question. Should be empty if the question can be answered directly by an agent. Multiple needs are independent from each other. If there is a dependence between the needs, they should be nested accordingly to reflect the dependence relationship. ",
                ),
            ),
        )
        QueryReplanModel.model_rebuild()
        return QueryReplanModel

    @staticmethod
    def map_query_replan_model_to_query_plan(
        query_replan_model: Any, id_index: dict[str, list["Need"]]
    ) -> "QueryPlan":
        query_plan = QueryPlan(
            id=query_replan_model.id
            if query_replan_model.id is not None
            else str(uuid.uuid4()),
            need=query_replan_model.need,
            question=query_replan_model.question,
            from_agent=query_replan_model.from_agent,
            to_agent=query_replan_model.to_agent,
            needs=[
                QueryPlan.map_query_replan_model_to_query_plan(need, id_index)
                for need in query_replan_model.needs
            ]
            if query_replan_model.needs
            else None,
            answered_needs=None,  # restored from id_index after mapping
        )

        QueryPlan.restore_answered_needs(query_plan, id_index)
        return query_plan

    def to_span_data(self) -> JsonValue:
        return {
            "id": self.id,
            "need": self.need,
            "question": self.question,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "needs": [need.to_span_data() for need in self.needs or []],
            "answered_needs": [
                answered_need.to_span_data()
                for answered_need in self.answered_needs or []
            ],
        }

    @staticmethod
    def replace_node(
        plan: "QueryPlan", node_id: str, replacement: "QueryPlan"
    ) -> "QueryPlan":
        """Return a new plan tree with the node matching node_id replaced by replacement."""
        if plan.id == node_id:
            return replacement
        if plan.needs:
            new_needs = [
                QueryPlan.replace_node(child, node_id, replacement)
                for child in plan.needs
            ]
            return plan.model_copy(update={"needs": new_needs})
        return plan


class ReplanMode(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"


class ReplanningNeeded(BaseModel):
    query_node: QueryPlan

    def to_span_data(self) -> JsonValue:
        return {"query_node": self.query_node.to_span_data()}


class ResponsibleAgentNotFound(ReplanningNeeded):
    pass


class NeedNotMet(ReplanningNeeded):
    partial_answer: PartialQueryAnswer

    def to_span_data(self) -> JsonValue:
        return {
            "query_node": self.query_node.to_span_data(),
            "partial_answer": self.partial_answer.to_span_data(),
        }


# Adapted from [RLM](https://github.com/alexzhang13/rlm), Original source file: rlm/core/types.py
def serialize_value(value: Any) -> JsonValue:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return serialize_value(list(value))
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
