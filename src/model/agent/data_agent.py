from typing import Hashable

from pydantic import BaseModel, ConfigDict


class DataAgent(BaseModel, Hashable):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str

    def __hash__(self):
        return hash((self.name, self.path))
