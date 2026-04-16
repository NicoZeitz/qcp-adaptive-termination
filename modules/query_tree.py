from dataclasses import dataclass
from typing import List, Literal

from pydantic import BaseModel, PrivateAttr

@dataclass
class Need(BaseModel):
    need: str
    answer: str

class QueryPlan(BaseModel):
    need: str | None
    question: str
    from_agent: Literal["User", "Coordinator"] | str
    to_agent: Literal["Coordinator"] | str
    needs: List["QueryPlan"] | None
    _answered_needs: List[Need] | None = PrivateAttr()
