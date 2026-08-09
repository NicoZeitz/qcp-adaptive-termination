from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models import result_type
from qcp_adaptive_termination.models.result_type import SerializedResultType


class AdditionalInformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    dependency: str
    data: SerializedResultType

    @staticmethod
    def preview_additional_information(
        additional_information: list["AdditionalInformation"],
    ) -> str | None:
        if not additional_information:
            return None
        parts = []
        for info in additional_information:
            data = info.data
            preview = result_type.serialize(data, context="preview") if result_type.is_result_type(data) else str(data)
            parts.append(f"step_id={info.step_id} | dependency={info.dependency}\n{preview}")
        return "\n\n".join(parts)
