"""
Agent selection strategy types.

Defines `SelectionStrategyType` (ranking algorithm enum) and `AgentSelection`
(agent + score pair used during multi-agent iteration).
"""

from enum import StrEnum


class SelectionStrategyType(StrEnum):
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

    # Posterior (embedding prior + sketch likelihood)
    POSTERIOR = "posterior"  # Needs: Table, Column
