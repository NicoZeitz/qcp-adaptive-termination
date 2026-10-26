"""Models for a drafted query plan and its field definitions."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.query_plan.query_plan_step import QueryPlanStep


class FieldDefinitions:
    """Reusable field descriptions for drafting a query plan."""

    dependency_step_id = Field(description="The step_id of the step that this step depends on.")
    dependency_step = Field(
        description=(
            "The step that this step depends on. This is used to access the answer of the dependent step when"
            " executing the query plan. This field is not expected to be provided by the user, but is filled in"
            " based on the provided dependency_step_id when parsing the query plan."
        )
    )
    dependency_description = Field(
        description="A natural language description of the output from the dependent step that is needed for this step."
    )

    step_id = Field(description="Unique identifier for the step. This is used to specify dependencies between steps.")
    step_query = Field(description="The query to be executed in this step.")
    step_to_agents = Field(
        description=(
            "The agent or agents to which the query is directed. This can either be a single agent or multiple"
            " agents. Agent names can be specific data agents or 'Coordinator' in case multiple dependencies"
            " should be aggregated in a way."
        )
    )
    step_merge_task = Field(
        description="A natural language instruction for merging the outputs of multiple agents in this step."
    )
    step_dependencies = Field(
        description="List of dependencies for this step that are required before this step can be executed."
    )
    step_answer = Field(description="The answer produced by this step, if available")

    plan_query = Field(description="The original user query for which this query plan was drafted.")
    plan_steps = Field(
        description=(
            "The steps of the query plan. Each step represents a query to be executed by an agent."
            " The steps should be ordered in the way they should be executed. The last step should produce"
            " the final answer to the user's query. Each step can specify dependencies on previous steps,"
            " which indicates that the step requires the output of those steps in order to execute its query."
        )
    )


class QueryPlan(BaseModel):
    """A plan of steps to answer the user's query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original user query.")
    steps: list[QueryPlanStep] = Field(description="The ordered steps of the plan.")
