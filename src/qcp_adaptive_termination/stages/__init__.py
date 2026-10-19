"""
Pipeline stage entry points.

Exposes the three main stages: discovery (`discovery_stage`),
planning (`planning_stage` / `replan_query_plan`), and execution
(`execute_query_plan`).
"""

from qcp_adaptive_termination.stages.discovery import (
    discovery_stage,
    filter_relevant_agents,
)
from qcp_adaptive_termination.stages.execution import execute_query_plan
from qcp_adaptive_termination.stages.planning import draft_query_plan, planning_stage
from qcp_adaptive_termination.stages.synthesis import synthesize_result

__all__ = [
    "discovery_stage",
    "draft_query_plan",
    "execute_query_plan",
    "filter_relevant_agents",
    "planning_stage",
    "synthesize_result",
]
