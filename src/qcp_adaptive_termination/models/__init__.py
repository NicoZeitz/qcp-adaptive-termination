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
from qcp_adaptive_termination.models.need import (
    NeedType,
    deserialize_need,
    exception_to_need,
    serialize_need,
    try_parse_value_to_need_type,
)
from qcp_adaptive_termination.models.result import (
    ResultType,
    get_failed_result,
    try_parse_value_to_result_type,
)
from qcp_adaptive_termination.models.selection_strategy import (
    AgentSelection,
    SelectionType,
)
from qcp_adaptive_termination.models.serializers import SerializedOptionalResultType

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
    # need
    "NeedType",
    "deserialize_need",
    "exception_to_need",
    "serialize_need",
    "try_parse_value_to_need_type",
    # result
    "ResultType",
    "get_failed_result",
    "try_parse_value_to_result_type",
    # selection_strategy
    "AgentSelection",
    "SelectionType",
    # serializers
    "SerializedOptionalResultType",
]
