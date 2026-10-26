"""Model for a dependency on another query plan step's result."""

from pydantic import BaseModel, ConfigDict, Field


class QueryDependency(BaseModel):
    """Dependency on another query plan step's result."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(description="The step this dependency belongs to.")
    dependency: str = Field(description="Identifier of the step depended upon.")
