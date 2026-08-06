from enum import Enum

from agents import MultiProvider
from openai.types.shared import Reasoning

from model.agent.agent_configuration import AgentConfiguration

OPENAI_MODEL = "openai/gpt-5"
UNIVERSITY_MODEL = "university/unsloth/DeepSeek-V4-Flash-0731"
WORK_MODEL = "work/gpt-5"
EMBEDDING_MODEL = "openai/text-embedding-3-large"

_multi_provider: MultiProvider | None = None


def set_multi_provider(provider: MultiProvider) -> None:
    global _multi_provider
    _multi_provider = provider


def get_multi_provider() -> MultiProvider:
    if _multi_provider is None:
        raise RuntimeError("MultiProvider not initialized — call initialize() first")
    return _multi_provider


class AgentRole(Enum):
    COORDINATOR_RELEVANCE = "Coordinator - Relevance"
    COORDINATOR_PLANNING = "Coordinator - Planning"
    COORDINATOR_CODE = "Coordinator - Code"
    COORDINATOR_REPLANNING = "Coordinator - Replanning"
    COORDINATOR_ALIGN_SCHEMA = "Coordinator - Align Schema"
    COORDINATOR_COMPLETENESS_EVALUATOR = "Coordinator - Completeness Evaluator"
    DATA_AGENT_SCHEMA_DESCRIPTION = "Data Agent - Schema Description"
    DATA_AGENT_RELEVANCE = "Data Agent - Relevance"
    DATA_AGENT_CODE = "Data Agent - Code"


_role_config: dict[AgentRole, tuple[str, Reasoning]] = {
    AgentRole.COORDINATOR_RELEVANCE: (UNIVERSITY_MODEL, Reasoning(effort="none")),
    AgentRole.COORDINATOR_PLANNING: (
        OPENAI_MODEL,
        Reasoning(effort="high", summary="concise"),
    ),
    AgentRole.COORDINATOR_CODE: (UNIVERSITY_MODEL, Reasoning(effort="high")),
    AgentRole.COORDINATOR_REPLANNING: (UNIVERSITY_MODEL, Reasoning(effort="none")),
    AgentRole.COORDINATOR_ALIGN_SCHEMA: (UNIVERSITY_MODEL, Reasoning(effort="high")),
    AgentRole.COORDINATOR_COMPLETENESS_EVALUATOR: (
        UNIVERSITY_MODEL,
        Reasoning(effort="high"),
    ),
    AgentRole.DATA_AGENT_SCHEMA_DESCRIPTION: (
        UNIVERSITY_MODEL,
        Reasoning(effort="none"),
    ),
    AgentRole.DATA_AGENT_RELEVANCE: (UNIVERSITY_MODEL, Reasoning(effort="none")),
    AgentRole.DATA_AGENT_CODE: (UNIVERSITY_MODEL, Reasoning(effort="high")),
}


def get_agent_configuration(
    role: AgentRole, agent_name: str | None = None
) -> AgentConfiguration:
    model, reasoning = _role_config[role]
    return AgentConfiguration(
        role=role.value,
        agent_name=agent_name,
        model=model,
        reasoning=reasoning,
    )
