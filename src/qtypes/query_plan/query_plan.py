from typing import Any, Literal
from dataclasses import dataclass

from pydantic import Field, JsonValue

from qtypes.query_plan.result import ResultType, result_to_span_data


class FieldDefinitions:
    dependency_step_id = Field(
        description="The step_id of the step that this step depends on."
    )
    dependency_step = Field(
        description="The step that this step depends on. This is used to access the answer of the dependent step when executing the query plan. This field is not expected to be provided by the user, but is filled in based on the provided dependency_step_id when parsing the query plan."
    )
    dependency_description = Field(
        description="A natural language description of the output from the dependent step that is needed for this step."
    )

    step_id = Field(
        description="Unique identifier for the step. This is used to specify dependencies between steps."
    )
    step_query = Field(description="The query to be executed in this step.")
    step_to_agents = Field(
        description="The agent to which the query is directed. Normally this is the name of a specific agent. However, it can also be 'Coordinator', which indicates that the query should be executed by the coordinator itself. This can be used for steps that involve reasoning or other operations that do not require access to a specific data source, instead only combining dependencies. The last step should always be directed to the Coordinator, as it is responsible to merge results and produce the final answer to the user query."
    )
    step_dependencies = Field(
        description="List of dependencies for this step that are required before this step can be executed."
    )
    step_answer = Field(description="The answer produced by this step, if available")

    plan_query = Field(
        description="The original user query for which this query plan was drafted."
    )
    plan_steps = Field(
        description="The steps of the query plan. Each step represents a query to be executed by an agent. The steps should be ordered in the way they should be executed. The last step should produce the final answer to the user's query. Each step can specify dependencies on previous steps, which indicates that the step requires the output of those steps in order to execute its query."
    )


@dataclass
class QueryDependency:
    step: "QueryPlanStep"
    dependency: str

    def to_span_data(self) -> JsonValue:
        return {"step_id": self.step.id, "dependency": self.dependency}

    @staticmethod
    def load_from_json(data: Any, steps_by_id: dict[str, "QueryPlanStep"]) -> "QueryDependency":
        step_id = data["step_id"]
        dependency = data["dependency"]

        step = steps_by_id.get(step_id)
        if step is None:
            raise ValueError(f"Step with id {step_id} not found in query plan steps")

        return QueryDependency(step=step, dependency=dependency)

@dataclass
class QueryPlanStep:
    id: str
    query: str
    to_agent: Literal["Coordinator"] | str
    dependencies: list[QueryDependency]
    answer: ResultType | None

    def to_span_data(self) -> JsonValue:
        return {
            "id": self.id,
            "query": self.query,
            "to_agent": self.to_agent,
            "dependencies": [
                dependency.to_span_data() for dependency in self.dependencies
            ],
            "answer_available": result_to_span_data(self.answer)
            if self.answer is not None
            else None,
        }

    @staticmethod
    def load_from_json(data: Any, steps_by_id: dict[str, "QueryPlanStep"]) -> "QueryPlanStep":
        id = data["id"]
        query = data["query"]
        to_agent = data["to_agent"]
        answer = data.get("answer")

        dependencies = []
        for dependency_data in data.get("dependencies", []):
            dependency = QueryDependency.load_from_json(dependency_data, steps_by_id)
            dependencies.append(dependency)

        return QueryPlanStep(
            id=id,
            query=query,
            to_agent=to_agent,
            dependencies=dependencies,
            answer=answer
        )


@dataclass
class QueryPlan:
    query: str
    steps: list[QueryPlanStep]

    def to_span_data(self) -> JsonValue:
        return {"steps": [step.to_span_data() for step in self.steps]}

    @staticmethod
    def load_from_json(data: Any) -> "QueryPlan":
        query = data["query"]
        steps_by_id = {}
        steps = []
        for step_data in data["steps"]:
            step = QueryPlanStep.load_from_json(step_data, steps_by_id)
            steps_by_id[step.id] = step
            steps.append(step)
        return QueryPlan(query=query, steps=steps)