"""
Pipeline stage entry points.

Exposes the three main stages: discovery (`filter_relevant_agents`),
planning (`draft_query_plan` / `replan_query_plan`), and execution
(`execute_query_plan`).
"""

from qcp_adaptive_termination.stages.discovery import DISCOVERY_STAGE_NAME, filter_relevant_agents
from qcp_adaptive_termination.stages.execution import STAGE_NAME as EXECUTION_STAGE_NAME
from qcp_adaptive_termination.stages.execution import execute_query_plan
from qcp_adaptive_termination.stages.planning import STAGE_NAME as PLANNING_STAGE_NAME
from qcp_adaptive_termination.stages.planning import draft_query_plan
from qcp_adaptive_termination.stages.replanning import replan_query_plan

__all__ = [
    "DISCOVERY_STAGE_NAME",
    "EXECUTION_STAGE_NAME",
    "PLANNING_STAGE_NAME",
    "draft_query_plan",
    "execute_query_plan",
    "filter_relevant_agents",
    "replan_query_plan",
]
