"""Selection strategy evaluation: benchmark ranking quality of each strategy."""

from scripts.selection_strategy_evaluation.aggregate import aggregate
from scripts.selection_strategy_evaluation.worker import run_query

INPUT_DIR = "input"
"""Dataset ``_meta`` sub-folder with the query inputs"""

__all__ = [
    "INPUT_DIR",
    "aggregate",
    "run_query",
]
