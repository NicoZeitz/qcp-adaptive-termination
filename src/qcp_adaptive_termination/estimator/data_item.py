from dataclasses import dataclass


@dataclass(slots=True)
class DataItem:
    """
    A single observed entity from one data source.

    Represents one row in an integrated database: the entity's canonical name
    (after entity resolution), its attribute value, and which source it came from.
    The same real-world entity may appear as multiple DataItems across sources —
    deduplication happens in Estimator via the name field.
    """

    name: str
    """Canonical entity identifier used to detect duplicates across sources."""

    value: float
    """Attribute value for this entity (e.g. number of employees, revenue)."""

    source_id: str
    """Identifier of the data source this item was observed in."""
