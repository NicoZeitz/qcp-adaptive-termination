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
    GENERATED_NL = "generated_nl"


class AgentRole(StrEnum):
    COORDINATOR_RELEVANCE = "coordinator_relevance"
    COORDINATOR_PLANNING = "coordinator_planning"
    COORDINATOR_CODE = "coordinator_code"
    COORDINATOR_REPLANNING = "coordinator_replanning"
    COORDINATOR_ALIGN_SCHEMA = "coordinator_align_schema"
    COORDINATOR_COMPLETENESS_EVALUATOR = "coordinator_completeness_evaluator"
    DATA_AGENT_SCHEMA_DESCRIPTION = "data_agent_schema_description"
    DATA_AGENT_RELEVANCE = "data_agent_relevance"
    DATA_AGENT_CODE = "data_agent_code"
    QUERY_DECOMPOSER = "query_decomposer"
    EMBEDDING_TEXT_GENERATOR = "embedding_text_generator"
    EMBEDDING_COLUMN_RELEVANCE = "embedding_column_relevance"
    EMBEDDING_ITERATIVE_RELEVANCE = "embedding_iterative_relevance"
    EMBEDDING_BOUNDARY_FINDER = "embedding_boundary_finder"
    EMBEDDING_BOUNDARY_FINDER_SCHEMA = "embedding_boundary_finder_schema"
    EMBEDDING_BOUNDARY_RECURSIVE = "embedding_boundary_recursive"
    EMBEDDING_BOUNDARY_RECURSIVE_AGENT = "embedding_boundary_recursive_agent"
    EMBEDDING_BOUNDARY_RECURSIVE_AGGREGATE = "embedding_boundary_recursive_aggregate"
    EMBEDDING_BOUNDARY_RECURSIVE_V3_AGENT = "embedding_boundary_recursive_v3_agent"
    EMBEDDING_BOUNDARY_RECURSIVE_V3_AGGREGATE = "embedding_boundary_recursive_v3_aggregate"
    BASELINE_FILTER = "baseline_filter"

    @property
    def prompt_name(self) -> str:
        return {
            AgentRole.COORDINATOR_RELEVANCE: "discovery-coordinator-relevance",
            AgentRole.COORDINATOR_PLANNING: "planning-coordinator-planning",
            AgentRole.COORDINATOR_CODE: "execution-coordinator-aggregate",
            AgentRole.COORDINATOR_REPLANNING: "execution-coordinator-replanning",
            AgentRole.COORDINATOR_ALIGN_SCHEMA: "execution-coordinator-align-schema",
            AgentRole.COORDINATOR_COMPLETENESS_EVALUATOR: "execution-coordinator-completeness-evaluator",
            AgentRole.DATA_AGENT_SCHEMA_DESCRIPTION: "discovery-data-agent-sql-schema-description",
            AgentRole.DATA_AGENT_RELEVANCE: "discovery-data-agent-sql-relevance",
            AgentRole.DATA_AGENT_CODE: "execution-data-agent-sql-code",
            AgentRole.QUERY_DECOMPOSER: "discovery-query-decomposer",
            AgentRole.EMBEDDING_TEXT_GENERATOR: "discovery-embedding-text-generator",
            AgentRole.EMBEDDING_COLUMN_RELEVANCE: "discovery-embedding-column-relevance",
            AgentRole.EMBEDDING_ITERATIVE_RELEVANCE: "discovery-embedding-iterative-relevance",
            AgentRole.EMBEDDING_BOUNDARY_FINDER: "discovery-embedding-boundary-finder",
            AgentRole.EMBEDDING_BOUNDARY_FINDER_SCHEMA: "discovery-embedding-boundary-finder-schema",
            AgentRole.EMBEDDING_BOUNDARY_RECURSIVE: "discovery-embedding-boundary-recursive",
            AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_AGENT: "discovery-embedding-boundary-recursive-agent",
            AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_AGGREGATE: "discovery-embedding-boundary-recursive-aggregate",
            AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_V3_AGENT: "discovery-embedding-boundary-recursive-v3-agent",
            AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_V3_AGGREGATE: "discovery-embedding-boundary-recursive-v3-aggregate",
            AgentRole.BASELINE_FILTER: "discovery-baseline-filter",
        }[self]
