"""
Stage-level result models.

Holds the output types for each pipeline stage: `FilteredRelevantAgents`
(discovery), `DraftedQueryPlan` (planning), and
`CollaborativeQueryAnsweringResult` (execution).
"""

from qcp_adaptive_termination.models.stages.collaborative_query_answering import (
    CollaborativeQueryAnswering,
)
from qcp_adaptive_termination.models.stages.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from qcp_adaptive_termination.models.stages.drafted_query_plan import DraftedQueryPlan
from qcp_adaptive_termination.models.stages.filtered_relevant_agents import (
    FilteredRelevantAgents,
)

__all__ = [
    "CollaborativeQueryAnswering",
    "CollaborativeQueryAnsweringResult",
    "DraftedQueryPlan",
    "FilteredRelevantAgents",
]
