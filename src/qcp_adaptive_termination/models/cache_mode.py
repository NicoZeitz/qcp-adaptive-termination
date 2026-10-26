"""
Cache operation modes.
"""

from enum import StrEnum


class CacheMode(StrEnum):
    """The read/write behavior of the cache."""

    READ_AND_WRITE = "READ_AND_WRITE"
    READ_AND_WRITE_THROW_ON_MISS = "READ_AND_WRITE_THROW_ON_MISS"
    READ_ONLY = "READ_ONLY"
    READ_ONLY_THROW_ON_MISS = "READ_ONLY_THROW_ON_MISS"
    WRITE_ONLY = "WRITE_ONLY"
    DISABLED = "DISABLED"
