from enum import StrEnum


class ValueEstimator(StrEnum):
    """Strategy for estimating the attribute values of missing entities."""

    NAIVE = "naive"
    """
    Mean substitution: assumes missing entities have the same average value
    as all observed entities (φ_K / c).
    """

    FREQUENCY = "frequency"
    """
    Singleton-mean substitution: uses the average value of singletons (φ_f1 / f1)
    as a proxy for missing entities, which are most similar to rarely seen ones.
    """

    FREQUENCY_F12 = "frequency_f12"
    """
    Singleton+doubleton-mean substitution: averages over singletons and doubletons
    (φ_f12 / (f1 + f2)) to reduce variance compared to singletons alone.
    """

    GOOD_TURING = "good_turing"
    """
    Good-Turing frequency smoothing: derives Δ directly from φ_f1 and the ratio
    f1 / (n - f1) without a separate count estimator.
    """
