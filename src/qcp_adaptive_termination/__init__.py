"""
Top-level package for qcp-adaptive-termination.

Exposes the main pipeline entry points, configuration, per-stage functions,
LLM provider setup, caching, and tracing/logging utilities.
"""

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.cache import setup_cache
from qcp_adaptive_termination.llm import model_provider
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.observability import (
    setup_logging,
    setup_tracing,
)
from qcp_adaptive_termination.pipeline import run_pipeline, run_query
from qcp_adaptive_termination.stages.discovery import filter_relevant_agents
from qcp_adaptive_termination.stages.execution import execute_query_plan
from qcp_adaptive_termination.stages.planning import draft_query_plan

__all__ = [
    "run_query",
    "run_pipeline",
    "Config",
    "filter_relevant_agents",
    "draft_query_plan",
    "execute_query_plan",
    "setup_logging",
    "setup_cache",
    "model_provider",
    "setup_tracing",
    "hooks",
]
