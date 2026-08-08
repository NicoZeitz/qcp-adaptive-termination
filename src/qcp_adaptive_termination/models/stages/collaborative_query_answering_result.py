from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.serializers import SerializedResultType


class CollaborativeQueryAnsweringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedResultType
