from enum import StrEnum


class CountEstimator(StrEnum):
    """Strategy for estimating the total number of unique entities in the population."""

    CHAO92 = "chao92"
    """Coverage-based estimator corrected for skew in the publicity distribution via CV²."""

    CHAO84 = "chao84"
    """Estimator based on the ratio of singletons to doubletons (f1² / 2·f2)."""

    JACKKNIFE1 = "jackknife1"
    """First-order jackknife: adds the singleton count f1 as a bias correction."""

    JACKKNIFE2 = "jackknife2"
    """Second-order jackknife: adds 2·f1 and subtracts f2 for a lower-bias estimate."""

    ACE = "ace"
    """
    Abundance-based coverage estimator: splits entities into rare/abundant groups
    and applies coverage correction only to the rare group.
    """

    UNSEEN = "unseen"
    """
    Linear-programming estimator that fits a non-parametric frequency distribution
    and extrapolates the number of unseen species.
    """
