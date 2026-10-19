"""
Statistical estimators for adaptive query termination.

Implements histogram-based count and value estimators (`CountEstimator`,
`ValueEstimator`) over `DataItem` / `Bucket` primitives, with configurable
bound types for confidence-interval queries.
"""

from qcp_adaptive_termination.estimator.data_item import DataItem
from qcp_adaptive_termination.estimator.estimator import Estimator
from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.models.estimation import (
    ConfidenceIntervalType,
    CountEstimator,
    ValueEstimator,
)

__all__ = [
    "ConfidenceIntervalType",
    "CountEstimator",
    "DataItem",
    "Estimator",
    "FrequencyStats",
    "ValueEstimator",
]
