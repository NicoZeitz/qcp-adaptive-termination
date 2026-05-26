from pydantic import BaseModel, ConfigDict, Field

from qtypes.query_plan_step import QueryPlanStep


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


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    steps: list[QueryPlanStep]
