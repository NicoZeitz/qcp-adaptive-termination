from pathlib import Path
from typing import Any

import yaml

from qcp_adaptive_termination.model.execution_config import ExecutionConfig


def load_config(source: str | Path | None) -> ExecutionConfig:
    if source is None:
        return ExecutionConfig()

    path = Path(source)
    if path.exists():
        raw: Any = yaml.safe_load(path.read_text())
    else:
        raw = yaml.safe_load(str(source))

    return ExecutionConfig.model_validate(raw)


def load_input(path: str | Path) -> Any:
    from qcp_adaptive_termination.cache import load_cache_file
    from qcp_adaptive_termination.model.stages.collaborative_query_answering import (
        CollaborativeQueryAnswering,
    )
    from qcp_adaptive_termination.model.stages.collaborative_query_answering_result import (
        CollaborativeQueryAnsweringResult,
    )
    from qcp_adaptive_termination.model.stages.drafted_query_plan import (
        DraftedQueryPlan,
    )
    from qcp_adaptive_termination.model.stages.filtered_relevant_agents import (
        FilteredRelevantAgents,
    )

    data, extra = load_cache_file(
        Path(path),
        deserializable_classes=[
            CollaborativeQueryAnswering,
            FilteredRelevantAgents,
            DraftedQueryPlan,
            CollaborativeQueryAnsweringResult,
        ],
    )
    return data, extra


def load_stage_input(path: str | Path, stage: int) -> Any:
    data, extra = load_input(path)
    return data, extra
