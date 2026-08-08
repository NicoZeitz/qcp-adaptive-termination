from enum import StrEnum


class ConfidenceIntervalType(StrEnum):
    """Method used to compute confidence intervals for the count estimate."""

    GOOD_TURING = "good_turing"
    """Bounds the missing mass M0 with an additive error term and converts to N via N = c / (1 - M0)."""

    CHAO84 = "chao84"
    """Asymmetric log-normal confidence interval for N - c; tighter than the Good-Turing bound.
    Falls back to Good-Turing when there are no doubletons."""
