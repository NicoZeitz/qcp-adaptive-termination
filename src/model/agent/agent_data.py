from dataclasses import dataclass, field


@dataclass
class AgentData:
    technical_description: str
    natural_language_description: str
    query_capabilities: list[str]
    embedding_descriptions: list[str]
    table_column_names: dict[str, dict[str, str]] = field(default_factory=dict)
