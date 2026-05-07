from dataclasses import dataclass

from pydantic import JsonValue

from qtypes.query_plan.result import ResultType, result_to_span_data
from utils.preview import CHAR_LIMIT_SPAN, preview_text


@dataclass
class AdditionalInformation:
    dependency: str
    data: ResultType

    def to_span_data(self) -> JsonValue:
        return {
            "dependency": preview_text(self.dependency, CHAR_LIMIT_SPAN),
            "data": result_to_span_data(self.data) if self.data is not None else None,
        }
