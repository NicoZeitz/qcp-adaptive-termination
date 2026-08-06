from dataclasses import dataclass


@dataclass
class ColumnDescription:
    column_name: str  # Name of the column
    natural_language_description: str  # Natural language description of the column
    technical_description: str  # MSchema


@dataclass
class TableDescription:
    table_name: str  # Name of the table
    natural_language_description: str  # Natural language description of the table
    technical_description: str  # MSchema
    columns: dict[str, ColumnDescription]  # Column descriptions, key is column name


@dataclass
class AgentRegistryData:
    natural_language_description: str  # Natural language description of the database
    technical_description: str  # MSchema string
    query_capabilities: list[
        str
    ]  # Query capabilities of the agent, e.g., ["Text-to-SQL"]
    individual_descriptions: dict[
        str, TableDescription
    ]  # Individual table descriptions, key is table name
