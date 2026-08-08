"""
Agent-level data models.

Defines `DataAgent`, `AgentConfiguration`, `AgentRegistryData` (with
`TableDescription` / `ColumnDescription`), and `AgentRegistryType`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Literal

from agents.run_config import DEFAULT_MAX_TURNS
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict


class DataAgent(BaseModel, Hashable):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sql"] = "sql"
    name: str
    path: str

    def __hash__(self):
        return hash((self.name, self.path))


class AgentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    agent_name: str | None = None
    model: str | None
    reasoning: Reasoning
    verbosity: Literal["low", "medium", "high"] | None = None
    max_turns: int | None = DEFAULT_MAX_TURNS * 2

    @property
    def display_name(self) -> str:
        return f"{self.agent_name} - {self.role}" if self.agent_name else self.role


@dataclass
class ColumnDescription:
    column_name: str
    natural_language_description: str
    technical_description: str


@dataclass
class TableDescription:
    table_name: str
    natural_language_description: str
    technical_description: str
    columns: dict[str, ColumnDescription]


@dataclass
class AgentRegistryData:
    natural_language_description: str
    technical_description: str
    query_capabilities: list[str]
    individual_descriptions: dict[str, TableDescription]


class AgentRegistryType(Enum):
    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"
