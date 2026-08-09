"""
Agent selection strategies and `AgentSelector`.

Implements multiple ranking strategies (sequential, embedding-based,
count/distinct-count, sketch-based, posterior, null-coverage, value-range)
and `AgentSelector`, which pre-computes strategy data and ranks agents at
each iteration of the multi-agent loop.
"""
from qcp_adaptive_termination.selection_strategy.agent_selector import (
    AgentSelector,
)
from qcp_adaptive_termination.selection_strategy.count_strategy import (
    CountStrategy,
    DistinctCountStrategy,
)
from qcp_adaptive_termination.selection_strategy.embedding_strategy import (
    EmbeddingStrategy,
)
from qcp_adaptive_termination.selection_strategy.posterior_strategy import (
    PosteriorStrategy,
)
from qcp_adaptive_termination.selection_strategy.sequential_strategy import (
    SequentialStrategy,
)
from qcp_adaptive_termination.selection_strategy.sketch_strategy import (
    HllSketchStrategy,
    MinHashStrategy,
    ThetaSketchStrategy,
)
from qcp_adaptive_termination.selection_strategy.strategy import (
    SelectionStrategy,
    StaticSelectionStrategy,
)

__all__ = [
    "AgentSelector",
    "CountStrategy",
    "DistinctCountStrategy",
    "EmbeddingStrategy",
    "HllSketchStrategy",
    "MinHashStrategy",
    "PosteriorStrategy",
    "SelectionStrategy",
    "SequentialStrategy",
    "StaticSelectionStrategy",
    "ThetaSketchStrategy",
]
