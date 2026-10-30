"""
Agent-level data models.

Defines `DataAgent`, `AgentRegistryData` (with `TableDescription` /
`ColumnDescription`), `AgentRegistryType`, and `AgentRole`.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DataAgent(BaseModel, Hashable):
    """A data agent that can be queried by the system."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sql"] = Field(default="sql", description="The agent type, always sql")
    name: str = Field(description="Unique name of the agent")
    path: str = Field(description="Path to the agent's data source")

    def __hash__(self):
        """Return a hash based on the agent name and path."""
        return hash((self.name, self.path))


@dataclass(slots=True)
class ColumnDescription:
    """Description of a single column in a data agent's schema."""

    column_name: str
    """Name of the column"""
    natural_language_description: str
    """Natural language description of the column"""
    technical_description: str
    """Technical description of the column"""


@dataclass(slots=True)
class TableDescription:
    """Description of a single table in a data agent's schema."""

    table_name: str
    """Name of the table"""
    natural_language_description: str
    """Natural language description of the table"""
    technical_description: str
    """Technical description of the table"""
    columns: dict[str, ColumnDescription]
    """Per-column descriptions keyed by column name"""


@dataclass(slots=True)
class AgentRegistryData:
    """Descriptions describing a data agent and its tables."""

    natural_language_description: str
    """Natural language description of the data agent"""
    technical_description: str
    """Technical description of the data agent"""
    query_capabilities: list[str]
    """Query capabilities supported by the agent"""
    individual_descriptions: dict[str, TableDescription]
    """Per-table descriptions keyed by table name"""


class AgentRegistryType(StrEnum):
    """Whether the registry uses natural language or technical descriptions."""

    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"


class AgentRole(StrEnum):
    """The role an agent plays during a pipeline stage."""

    DISCOVERY_DATA_AGENT_SCHEMA_DESCRIPTION = "discovery-data-agent-schema-description"
    DISCOVERY_COORDINATOR_RELEVANCE = "discovery-coordinator-relevance"
    DISCOVERY_DATA_AGENT_RELEVANCE = "discovery-data-agent-relevance"
    BASELINE_FILTER = "discovery-baseline-filter"
    EMBEDDING_BOUNDARY_FINDER = "discovery-embedding-boundary-finder"
    ITERATIVE_EMBEDDING_FILTER_AGENT_RELEVANCE = "discovery-iterative-embedding-filter-agent-relevance"
    ITERATIVE_EMBEDDING_FILTER_QUERY_COMPLETENESS = "discovery-iterative-embedding-filter-query-completeness"
    ITERATIVE_EMBEDDING_FILTER_DECOMPOSER = "discovery-iterative-embedding-filter-decomposer"
    PLANNING_COORDINATOR_DRAFT_QUERY_PLAN = "planning-coordinator-draft-query-plan"
    EXECUTION_COORDINATOR_ALIGN_SCHEMA = "execution-coordinator-align-schema"
    EXECUTION_COORDINATOR_WORLD_KNOWLEDGE_STOP = "execution-coordinator-world-knowledge-stop"
    EXECUTION_COORDINATOR_RESULT_PRODUCER = "execution-coordinator-result-producer"
    EXECUTION_COORDINATOR_ANSWER_STEP = "execution-coordinator-answer-step"
    EXECUTION_DATA_AGENT_QUERY = "execution-data-agent-query"
    EXECUTION_COORDINATOR_REPLAN = "execution-coordinator-replan"
