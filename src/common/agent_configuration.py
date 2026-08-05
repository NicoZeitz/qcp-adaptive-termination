from agents import MultiProvider
from openai.types.shared import Reasoning

from model.agent.agent_configuration import AgentConfiguration
from model.agent.data_agent import DataAgent

OPENAI_MODEL = "openai/gpt-5"
# UNIVERSITY_MODEL = "university/unsloth/MiniMax-M3"
UNIVERSITY_MODEL = "university/unsloth/DeepSeek-V4-Flash-0731"
WORK_MODEL = "work/gpt-5"
# EMBEDDING_MODEL = "work/text-embedding-3-large"
EMBEDDING_MODEL = "openai/text-embedding-3-large"

_multi_provider: MultiProvider | None = None


def set_multi_provider(provider: MultiProvider) -> None:
    global _multi_provider
    _multi_provider = provider


def get_multi_provider() -> MultiProvider:
    if _multi_provider is None:
        raise RuntimeError("MultiProvider not initialized — call initialize() first")
    return _multi_provider


# ---------------------------------------------------------------------------
# Per-role coordinator config
# ---------------------------------------------------------------------------
_coordinator_config: dict[str, tuple[str, Reasoning]] = {
    "Default": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Relevance": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Planning": (OPENAI_MODEL, Reasoning(effort="high", summary="concise")),
    # "Planning": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Code": (UNIVERSITY_MODEL, Reasoning(effort="high")),
    # "Replanning": (OPENAI_MODEL, Reasoning(effort="high", summary="concise")),
    "Replanning": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Align Schema": (UNIVERSITY_MODEL, Reasoning(effort="high")),
    "Completeness Evaluator": (UNIVERSITY_MODEL, Reasoning(effort="high")),
}

# ---------------------------------------------------------------------------
# Per-role data agent config
# ---------------------------------------------------------------------------
_data_agent_config: dict[str, tuple[str, Reasoning]] = {
    "Default": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Schema Description": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Relevance": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Code": (UNIVERSITY_MODEL, Reasoning(effort="high")),
}


def get_coordinator_agent_configuration(name_postfix: str) -> AgentConfiguration:
    model, reasoning = _coordinator_config.get(
        name_postfix, _coordinator_config["Default"]
    )
    return AgentConfiguration(
        name=f"Query Coordinator - {name_postfix}",
        model=model,
        reasoning=reasoning,
    )


def get_data_agent_agent_configuration(
    data_agent: DataAgent, name_postfix: str
) -> AgentConfiguration:
    model, reasoning = _data_agent_config.get(
        name_postfix, _data_agent_config["Default"]
    )
    return AgentConfiguration(
        name=f"{data_agent.name} - {name_postfix}",
        model=model,
        reasoning=reasoning,
    )
