from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.result_type import SerializedResultType
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class SynthesizedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedResultType
    summary: str
    combined_completeness: float | None = None
    step_uncertainties: list[StepUncertainty] = Field(default_factory=list)
