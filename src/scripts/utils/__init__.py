"""Shared utilities for visualization scripts."""

from scripts.utils.evaluation import (
    EvaluationCurve,
    EvaluationResult,
    build_curve,
    compute_auc,
    compute_greedy_oracle,
)
from scripts.utils.evoc import (
    CATEGORICAL_COLORS,
    EVOC_BASE_HUES,
    NOISE_COLOR,
    build_color_map,
    build_evoc_layout,
)
from scripts.utils.geo import DB_GEO
from scripts.utils.setup import ScriptContext, run_script
from scripts.utils.sql import parse_select_columns, rewrite_sql_identifiers

__all__ = [
    "CATEGORICAL_COLORS",
    "DB_GEO",
    "EVOC_BASE_HUES",
    "EvaluationCurve",
    "EvaluationResult",
    "NOISE_COLOR",
    "ScriptContext",
    "build_color_map",
    "build_curve",
    "build_evoc_layout",
    "compute_auc",
    "compute_greedy_oracle",
    "parse_select_columns",
    "rewrite_sql_identifiers",
    "run_script",
]
