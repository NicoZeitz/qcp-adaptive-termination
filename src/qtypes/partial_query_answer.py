from typing import Literal

from pydantic import JsonValue, BaseModel

from qtypes.query_plan.need import NeedType, needs_to_span_data
from qtypes.query_plan.result import ResultType, result_to_span_data

class PartialQueryAnswerDummy(BaseModel):
    pass

class PartialQueryAnswer:
    answer: ResultType | None = None
    has_answer: bool = False
    needs: list[NeedType] = []
    type: Literal["answer", "needs", "answer_and_needs", "empty"] = "empty"

    def set_answer(self, answer: ResultType):
        self.answer = answer
        self.has_answer = True

    def add_needs(self, needs: set[NeedType]):
        self.needs = list(set(self.needs).union(needs))

    @property
    def is_final(self) -> bool:
        return self.answer is not None or len(self.needs) > 0

    @property
    def has_needs(self) -> bool:
        return len(self.needs) > 0

    def to_span_data(self) -> JsonValue:
        return {
            "answer": result_to_span_data(self.answer)
            if self.answer is not None
            else None,
            "needs": needs_to_span_data(self.needs),
        }
