"""
Pipeline stage entry points.

Exposes the three main stages: discovery (`filter_relevant_agents`),
planning (`draft_query_plan` / `replan_query_plan`), and execution
(`execute_query_plan`).
"""

from qcp_adaptive_termination.stages.discovery import (
    filter_relevant_agents,
    filter_relevant_agents_internal,
)
from qcp_adaptive_termination.stages.planning import draft_query_plan
from qcp_adaptive_termination.stages.replanning import replan_query_plan

__all__ = [
    "filter_relevant_agents",
    "filter_relevant_agents_internal",
    "draft_query_plan",
    "replan_query_plan",
]
