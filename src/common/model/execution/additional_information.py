from pydantic import BaseModel, ConfigDict

from common.model.serializers import SerializedResultType


class AdditionalInformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    dependency: str
    data: SerializedResultType
