"""
Stage-level result models.

Holds the output types for each pipeline stage: `FilteredRelevantAgents`
(discovery), `DraftedQueryPlan` (planning),
`CollaborativeQueryAnsweringResult` (execution), `SynthesizedResult`
(synthesis), plus the `Stage` enum.
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
from qcp_adaptive_termination.models.stages.stage import Stage
from qcp_adaptive_termination.models.stages.synthesized_result import SynthesizedResult

__all__ = [
    "CollaborativeQueryAnswering",
    "CollaborativeQueryAnsweringResult",
    "DraftedQueryPlan",
    "FilteredRelevantAgents",
    "Stage",
    "SynthesizedResult",
]
