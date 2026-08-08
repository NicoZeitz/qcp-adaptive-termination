"""
Agent configuration, registry, and LLM runner utilities.

Provides role-based agent configurations, a registry of available agents,
and functions to invoke agents or raw LLM calls.
"""

from qcp_adaptive_termination.agents.configuration import (
    AgentRole,
    get_agent_configuration,
)
from qcp_adaptive_termination.llm.runner import run_agent, run_llm
import qcp_adaptive_termination.agents.registry as registry

__all__ = [
    "AgentRole",
    "get_agent_configuration",
    "run_agent",
    "run_llm",
    "registry",
]
