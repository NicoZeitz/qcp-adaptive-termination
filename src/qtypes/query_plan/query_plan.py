from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from qtypes.query_plan.result import ResultType


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


class QueryDependency(BaseModel):
    step: "QueryPlanStep" = FieldDefinitions.dependency_step
    dependency: str = FieldDefinitions.dependency_description

    def to_span_data(self) -> JsonValue:
        return {"step_id": self.step.id, "dependency": self.dependency}


class QueryPlanStep(BaseModel):
    id: str = FieldDefinitions.step_id
    query: str = FieldDefinitions.step_query
    to_agent: Literal["Coordinator"] | str = FieldDefinitions.step_to_agents
    dependencies: list[QueryDependency] = FieldDefinitions.step_dependencies
    answer: ResultType | None = FieldDefinitions.step_answer

    def to_span_data(self) -> JsonValue:
        return {
            "id": self.id,
            "query": self.query,
            "to_agent": self.to_agent,
            "dependencies": [
                dependency.to_span_data() for dependency in self.dependencies
            ],
            "answer_available": self.answer is not None,
        }


class QueryPlan(BaseModel):
    query: str = FieldDefinitions.plan_query
    steps: list[QueryPlanStep] = FieldDefinitions.plan_steps

    def to_span_data(self) -> JsonValue:
        return {"steps": [step.to_span_data() for step in self.steps]}
