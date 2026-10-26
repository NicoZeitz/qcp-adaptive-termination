"""Pipeline stage identifiers."""

from enum import StrEnum


class Stage(StrEnum):
    """Identifier of a pipeline stage."""

    DISCOVERY = "discovery"
    PLANNING = "planning"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"

    @property
    def order(self) -> int:
        """Zero-based position of the stage in the pipeline."""
        return _STAGE_ORDER[self]

    @property
    def label(self) -> str:
        """Human-readable stage label, e.g. "Stage 1: Discovery"."""
        return f"Stage {self.order + 1}: {self.name.title()}"


_STAGE_ORDER: dict[Stage, int] = {
    Stage.DISCOVERY: 0,
    Stage.PLANNING: 1,
    Stage.EXECUTION: 2,
    Stage.SYNTHESIS: 3,
}
