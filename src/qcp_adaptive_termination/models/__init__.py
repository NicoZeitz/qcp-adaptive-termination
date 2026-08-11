"""
Shared data models and domain types.

Contains Pydantic/dataclass models for configuration, query plans, execution
results, needs, estimation, and agent definitions used across all pipeline
stages.
"""

import qcp_adaptive_termination.models.need_type as need_type
import qcp_adaptive_termination.models.result_type as result_type
from qcp_adaptive_termination.models.agent import (
    AgentRegistryData,
    AgentRegistryType,
    AgentRole,
    ColumnDescription,
    DataAgent,
    TableDescription,
)
from qcp_adaptive_termination.models.cache_mode import CacheMode
from qcp_adaptive_termination.models.config import (
    Config,
    RoleConfig,
)
from qcp_adaptive_termination.models.estimation import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
    EstimatorMode,
    EstimatorStatistics,
)
from qcp_adaptive_termination.models.need_type import (
    NeedType,
    SerializedNeedList,
)
from qcp_adaptive_termination.models.result_type import (
    ResultType,
    SerializedOptionalResultType,
    SerializedResultType,
)
from qcp_adaptive_termination.models.scored_data_agent import (
    ScoredDataAgent,
)
from qcp_adaptive_termination.models.selection_strategy_type import (
    SelectionStrategyType,
)

__all__ = [
    # agent
    "AgentRegistryData",
    "AgentRegistryType",
    "AgentRole",
    "ColumnDescription",
    "DataAgent",
    "TableDescription",
    # cache_mode
    "CacheMode",
    # config
    "Config",
    "RoleConfig",
    # estimation
    "AggregationType",
    "AlignedResultList",
    "AlignedSchemaMetadata",
    "EstimatorMode",
    "EstimatorStatistics",
    # need
    "need_type",
    "NeedType",
    "SerializedNeedList",
    # result
    "result_type",
    "ResultType",
    "SerializedResultType",
    "SerializedOptionalResultType",
    # selection_strategy
    "SelectionStrategyType",
    "ScoredDataAgent",
]
