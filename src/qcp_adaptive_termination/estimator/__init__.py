"""
Statistical estimators for adaptive query termination.

Implements histogram-based count and value estimators (`CountEstimator`,
`ValueEstimator`) over `DataItem` / `Bucket` primitives, with configurable
bound types for confidence-interval queries.
"""

from qcp_adaptive_termination.estimator.confidence_interval_type import (
    ConfidenceIntervalType,
)
from qcp_adaptive_termination.estimator.count_estimator import CountEstimator
from qcp_adaptive_termination.estimator.data_item import DataItem
from qcp_adaptive_termination.estimator.estimator import Estimator
from qcp_adaptive_termination.estimator.value_estimator import ValueEstimator

__all__ = [
    "ConfidenceIntervalType",
    "CountEstimator",
    "DataItem",
    "Estimator",
    "ValueEstimator",
]
