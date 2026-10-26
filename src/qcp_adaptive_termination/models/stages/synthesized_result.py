"""Result model for the synthesis stage."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.result_type import SerializedResultType
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class SynthesizedResult(BaseModel):
    """Final synthesized answer together with its completeness metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedResultType = Field(description="The final synthesized result.")
    summary: str = Field(description="Natural language summary of the answer.")
    combined_completeness: float | None = Field(
        default=None, description="Combined completeness across steps, if estimable."
    )
    step_uncertainties: dict[str, StepUncertainty] = Field(
        default_factory=dict, description="Recorded uncertainty per sub-step, keyed by step id."
    )
