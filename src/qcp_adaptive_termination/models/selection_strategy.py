"""
Agent selection strategy types.

Defines `SelectionType` (ranking algorithm enum) and `AgentSelection`
(agent + score pair used during multi-agent iteration).
"""

from dataclasses import dataclass
from enum import Enum

from qcp_adaptive_termination.models.agent import DataAgent


class SelectionType(Enum):
    SEQUENTIAL = "sequential"  # Needs: None

    # Embedding-based
    EMBEDDING = "embedding"  # Needs: None

    # Count-based
    COUNT = "count"  # Needs: Table
    DISTINCT_COUNT = "distinct_count"  # Needs: Table, Column

    # Sketch-based (point estimate)
    HLL_SKETCH = "hll_sketch"  # Needs: Table, Column
    THETA_SKETCH = "theta_sketch"  # Needs: Table, Column
    MIN_HASH = "min_hash"  # Needs: Table, Column

    # Sketch-based (upper confidence bound — optimistic ranking)
    HLL_SKETCH_UCB = "hll_sketch_ucb"  # Needs: Table, Column
    THETA_SKETCH_UCB = "theta_sketch_ucb"  # Needs: Table, Column

    # Posterior (embedding prior + sketch likelihood)
    POSTERIOR = "posterior"  # Needs: Table, Column


@dataclass
class AgentSelection:
    agent: DataAgent
    score: float | None = None
