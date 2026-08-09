"""
Shared data models and domain types.

Contains Pydantic/dataclass models for configuration, query plans, execution
results, needs, estimation, and agent definitions used across all pipeline
stages.
"""

from qcp_adaptive_termination.models.agent import (
    AgentRegistryData,
    AgentRegistryType,
    AgentRole,
    ColumnDescription,
    DataAgent,
    TableDescription,
)
from qcp_adaptive_termination.models.config import (
    Config,
    RoleConfig,
    _config_var,
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
    deserialize,
    from_exception,
    serialize,
    try_parse_value,
)
from qcp_adaptive_termination.models.result_type import (
    ResultType,
    SerializedOptionalResultType,
    SerializedResultType,
    failed_result,
)
from qcp_adaptive_termination.models.result_type import (
    try_parse_value as try_parse_result_value,
)
from qcp_adaptive_termination.models.selection_strategy import (
    AgentSelection,
    SelectionType,
)

__all__ = [
    # agent
    "AgentRegistryData",
    "AgentRegistryType",
    "AgentRole",
    "ColumnDescription",
    "DataAgent",
    "TableDescription",
    # config
    "Config",
    "RoleConfig",
    "_config_var",
    # estimation
    "AggregationType",
    "AlignedResultList",
    "AlignedSchemaMetadata",
    "EstimatorMode",
    "EstimatorStatistics",
    # need_type
    "NeedType",
    "SerializedNeedList",
    "deserialize",
    "from_exception",
    "serialize",
    "try_parse_value",
    # result_type
    "ResultType",
    "SerializedOptionalResultType",
    "SerializedResultType",
    "failed_result",
    "try_parse_result_value",
    # selection_strategy
    "AgentSelection",
    "SelectionType",
]
