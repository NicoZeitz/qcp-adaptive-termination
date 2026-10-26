"""Data item dataclass for a single observation of an entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class DataItem:
    """A single observation of an entity from one data source."""

    name: str
    """Canonical entity identifier used to detect duplicates across sources."""

    value: float
    """Attribute value for this entity (e.g. number of employees, revenue)."""

    source_id: str
    """Identifier of the data source this item was observed in."""
