"""Models for a query plan step's partial answer."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.need_type import NeedType, SerializedNeedList
from qcp_adaptive_termination.models.result_type import (
    ResultType,
    SerializedOptionalResultType,
)


class PartialQueryAnswer(BaseModel):
    """Result of a single step's answer logic"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    answer: SerializedOptionalResultType = Field(default=None, description="The step's answer, if any.")
    has_answer: bool = Field(default=False, description="Whether the step produced an answer.")
    needs: SerializedNeedList = Field(default=[], description="Missing information requested from the step.")

    def set_answer(self, answer: ResultType) -> Self:
        """Store the given answer and mark the step as answered."""
        self.answer = answer
        self.has_answer = True
        return self

    def add_needs(self, needs: set[NeedType] | list[NeedType]) -> Self:
        """Merge needs into the step's existing needs list."""
        self.needs = list(set(self.needs).union(needs))
        return self

    @property
    def has_needs(self) -> bool:
        """Return whether any needs are set."""
        return len(self.needs) > 0

    @property
    def is_final(self) -> bool:
        """Return whether the step produced an answer or needs."""
        return self.has_answer or self.has_needs
