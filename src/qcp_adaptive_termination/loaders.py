from pathlib import Path
from typing import Any

import yaml

from qcp_adaptive_termination.cache import load_cache_file
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
)


def load_config(source: str | Path | None) -> Config:
    if source is None:
        return Config()

    config_path = Path(source)
    raw_config = yaml.safe_load(config_path.read_text() if config_path.exists() else str(source))

    return Config.model_validate(raw_config)


def load_input(path: str | Path) -> Any:
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
