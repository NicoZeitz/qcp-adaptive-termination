"""
Agent-level data models.

Defines `DataAgent`, `AgentRegistryData` (with `TableDescription` /
`ColumnDescription`), `AgentRegistryType`, and `AgentRole`.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DataAgent(BaseModel, Hashable):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sql"] = "sql"
    name: str
    path: str

    def __hash__(self):
        return hash((self.name, self.path))


@dataclass(slots=True)
class ColumnDescription:
    column_name: str
    natural_language_description: str
    technical_description: str


@dataclass(slots=True)
class TableDescription:
    table_name: str
    natural_language_description: str
    technical_description: str
    columns: dict[str, ColumnDescription]


@dataclass(slots=True)
class AgentRegistryData:
    natural_language_description: str
    technical_description: str
    query_capabilities: list[str]
    individual_descriptions: dict[str, TableDescription]


class AgentRegistryType(StrEnum):
    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"


class AgentRole(StrEnum):
    DISCOVERY_DATA_AGENT_SCHEMA_DESCRIPTION = "discovery-data-agent-schema-description"
    DISCOVERY_COORDINATOR_RELEVANCE = "discovery-coordinator-relevance"
    DISCOVERY_DATA_AGENT_RELEVANCE = "discovery-data-agent-relevance"
    PLANNING_COORDINATOR_DRAFT_QUERY_PLAN = "planning-coordinator-draft-query-plan"
    EXECUTION_COORDINATOR_ALIGN_SCHEMA = "execution-coordinator-align-schema"
    EXECUTION_COORDINATOR_WORLD_KNOWLEDGE_STOP = "execution-coordinator-world-knowledge-stop"
    EXECUTION_COORDINATOR_RESULT_PRODUCER = "execution-coordinator-result-producer"
    EXECUTION_COORDINATOR_ANSWER_STEP = "execution-coordinator-answer-step"
    EXECUTION_DATA_AGENT_QUERY = "execution-data-agent-query"
    EXECUTION_COORDINATOR_REPLAN = "execution-coordinator-replan"
