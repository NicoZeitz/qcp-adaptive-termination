"""Completeness evaluation: measure early-stopping quality of the completeness evaluator."""

from scripts.completeness_evaluation.aggregate import aggregate
from scripts.completeness_evaluation.worker import run_query

INPUT_DIR = "input"
"""Dataset ``_meta`` sub-folder with the query inputs"""

__all__ = [
    "INPUT_DIR",
    "aggregate",
    "run_query",
]
