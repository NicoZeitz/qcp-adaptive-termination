"""
Shared data models and domain types.

Contains Pydantic/dataclass models for configuration, query plans, execution
results, needs, estimation, and agent definitions used across all pipeline
stages.
"""

from qcp_adaptive_termination.models.config import (
    Config,
    RoleConfig,
    config_context,
    config_var,
)
from qcp_adaptive_termination.models.need import (
    deserialize_need,
    exception_to_need,
    serialize_need,
    try_parse_value_to_need_type,
)
from qcp_adaptive_termination.models.result import (
    get_failed_result,
    try_parse_value_to_result_type,
)
from qcp_adaptive_termination.models.serializers import SerializedOptionalResultType

__all__ = [
    "Config",
    "RoleConfig",
    "config_context",
    "config_var",
    "deserialize_need",
    "exception_to_need",
    "serialize_need",
    "try_parse_value_to_need_type",
    "get_failed_result",
    "try_parse_value_to_result_type",
    "SerializedOptionalResultType",
]
