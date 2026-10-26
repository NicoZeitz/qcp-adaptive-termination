"""
Top-level package for qcp-adaptive-termination.

Exposes the main pipeline entry points, configuration, per-stage functions,
LLM provider setup, caching, and tracing/logging utilities.
"""

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.observability.context import (
    cache_context,
    config_context,
    data_context,
    logging_context,
    pipeline_context,
    tracing_context,
)
from qcp_adaptive_termination.pipeline import load_config, load_input, run_pipeline, run_query, run_stages
from qcp_adaptive_termination.stages.discovery import discovery_stage
from qcp_adaptive_termination.stages.execution import execute_query_plan
from qcp_adaptive_termination.stages.planning import draft_query_plan

__all__ = [
    "Config",
    "cache_context",
    "config_context",
    "data_context",
    "draft_query_plan",
    "execute_query_plan",
    "discovery_stage",
    "hooks",
    "load_config",
    "load_input",
    "logging_context",
    "pipeline_context",
    "run_pipeline",
    "run_query",
    "run_stages",
    "tracing_context",
]
