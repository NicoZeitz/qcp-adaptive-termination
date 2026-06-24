from agents import MultiProvider
from openai.types.shared import Reasoning

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent

OPENAI_MODEL = "openai/gpt-5"
UNIVERSITY_MODEL = "university/unsloth/MiniMax-M3"
WORK_MODEL = "work/gpt-5"

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
    "Code": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Replanning": (OPENAI_MODEL, Reasoning(effort="high", summary="concise")),
    "Align Schema": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Completeness Evaluator": (UNIVERSITY_MODEL, Reasoning(effort="none")),
}

# ---------------------------------------------------------------------------
# Per-role data agent config
# ---------------------------------------------------------------------------
_data_agent_config: dict[str, tuple[str, Reasoning]] = {
    "Default": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Schema Description": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Relevance": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Summary": (UNIVERSITY_MODEL, Reasoning(effort="none")),
    "Code": (UNIVERSITY_MODEL, Reasoning(effort="none")),
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
