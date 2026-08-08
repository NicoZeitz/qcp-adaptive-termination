from enum import Enum
from typing import cast, Literal

from openai.types.shared import Reasoning

from qcp_adaptive_termination.models.agent.agent_configuration import AgentConfiguration
from qcp_adaptive_termination.models.config import RoleConfig, config_var


class AgentRole(Enum):
    COORDINATOR_RELEVANCE = "coordinator_relevance"
    COORDINATOR_PLANNING = "coordinator_planning"
    COORDINATOR_CODE = "coordinator_code"
    COORDINATOR_REPLANNING = "coordinator_replanning"
    COORDINATOR_ALIGN_SCHEMA = "coordinator_align_schema"
    COORDINATOR_COMPLETENESS_EVALUATOR = "coordinator_completeness_evaluator"
    DATA_AGENT_SCHEMA_DESCRIPTION = "data_agent_schema_description"
    DATA_AGENT_RELEVANCE = "data_agent_relevance"
    DATA_AGENT_CODE = "data_agent_code"


def get_agent_configuration(
    role: AgentRole, agent_name: str | None = None
) -> AgentConfiguration:
    roles = config_var.get().agent_configs
    role_cfg: RoleConfig = roles[role.value]
    effort = cast(
        Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        role_cfg.reasoning_effort,
    )
    summary = cast(
        Literal["auto", "concise", "detailed"] | None,
        role_cfg.reasoning_summary,
    )
    reasoning = Reasoning(effort=effort, summary=summary)
    return AgentConfiguration(
        role=role.value,
        agent_name=agent_name,
        model=role_cfg.model,
        reasoning=reasoning,
    )
