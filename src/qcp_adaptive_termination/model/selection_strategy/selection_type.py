from enum import Enum


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

    # Statistics-based
    NULL_COVERAGE = "null_coverage"  # Needs: Table, Column
    VALUE_RANGE = "value_range"  # Needs: Table, Column

    # Posterior (embedding prior + sketch likelihood)
    POSTERIOR = "posterior"  # Needs: Table, Column
