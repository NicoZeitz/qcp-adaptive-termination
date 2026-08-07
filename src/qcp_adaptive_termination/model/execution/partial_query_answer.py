from typing import Self

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.model.need import NeedType
from qcp_adaptive_termination.model.serializers import (
    SerializedNeedList,
    SerializedOptionalResultType,
)
from qcp_adaptive_termination.model.result import ResultType


class PartialQueryAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    answer: SerializedOptionalResultType = None
    has_answer: bool = False
    needs: SerializedNeedList = []

    def set_answer(self, answer: ResultType) -> Self:
        self.answer = answer
        self.has_answer = True
        return self

    def add_needs(self, needs: set[NeedType] | list[NeedType]) -> Self:
        self.needs = list(set(self.needs).union(needs))
        return self

    @property
    def has_needs(self) -> bool:
        return len(self.needs) > 0

    @property
    def is_final(self) -> bool:
        return self.has_answer or self.has_needs
