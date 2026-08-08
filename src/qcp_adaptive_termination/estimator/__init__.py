"""
Statistical estimators for adaptive query termination.

Implements histogram-based count and value estimators (`CountEstimator`,
`ValueEstimator`) over `DataItem` / `Bucket` primitives, with configurable
bound types for confidence-interval queries.
"""

from .data_item import DataItem
from .bucket import Bucket, HistogramBar
from .estimator import BoundType, CountEstimator, Estimator, ValueEstimator

__all__ = [
    "HistogramBar",
    "DataItem",
    "Bucket",
    "BoundType",
    "CountEstimator",
    "ValueEstimator",
    "Estimator",
]
