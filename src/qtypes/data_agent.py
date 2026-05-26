from typing import Hashable, Literal

from pydantic import BaseModel, ConfigDict


class DataAgent(BaseModel, Hashable):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sql", "text"]
    name: str
    path: str

    def __hash__(self):
        return hash((self.type, self.name, self.path))
