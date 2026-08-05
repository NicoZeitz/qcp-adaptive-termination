from pydantic import BaseModel, ConfigDict


class QueryDependency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    dependency: str
