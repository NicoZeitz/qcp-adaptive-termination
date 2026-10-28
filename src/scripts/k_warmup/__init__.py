"""k-warmup study: find where the unknown-unknowns estimator stabilizes per ordering."""

from scripts.k_warmup.aggregate import aggregate
from scripts.k_warmup.worker import run_query

INPUT_DIR = "input_tables"
"""Dataset ``_meta`` sub-folder with one single-step SELECT * input per table (written by the dataset)"""

__all__ = [
    "INPUT_DIR",
    "aggregate",
    "run_query",
]
