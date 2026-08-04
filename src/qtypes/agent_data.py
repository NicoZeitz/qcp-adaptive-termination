from dataclasses import dataclass


@dataclass
class AgentData:
    technical_description: str
    natural_language_description: str
    query_capabilities: list[str]
    embedding_descriptions: list[str]
