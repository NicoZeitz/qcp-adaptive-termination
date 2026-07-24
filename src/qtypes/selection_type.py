from enum import Enum


class SelectionType(Enum):
    SEQUENTIAL = "sequential"

    # Count-based
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"

    # Embedding-based
    EMBEDDING = "embedding"
    EMBEDDING_MSCHEMA = "embedding_mschema"

    # Sketch-based (point estimate)
    HLL_SKETCH = "hll_sketch"
    THETA_SKETCH = "theta_sketch"
    MIN_HASH = "min_hash"

    # Sketch-based (upper confidence bound — optimistic ranking)
    HLL_SKETCH_UCB = "hll_sketch_ucb"
    THETA_SKETCH_UCB = "theta_sketch_ucb"

    # Statistics-based
    NULL_COVERAGE = "null_coverage"
    VALUE_RANGE = "value_range"
