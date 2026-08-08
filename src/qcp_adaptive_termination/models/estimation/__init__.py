"""
Estimation result models.

Holds `AlignedResultList` (schema-aligned query results with metadata),
`EstimatorMode`, and `EstimatorStatistics` used by the termination logic.
"""

from qcp_adaptive_termination.models.estimation.aligned_result_list import (
    AlignedResultList,
    AlignedSchemaMetadata,
)
from qcp_adaptive_termination.models.estimation.estimator_mode import EstimatorMode
from qcp_adaptive_termination.models.estimation.estimator_statistics import (
    EstimatorStatistics,
)

__all__ = [
    "AlignedResultList",
    "AlignedSchemaMetadata",
    "EstimatorMode",
    "EstimatorStatistics",
]
