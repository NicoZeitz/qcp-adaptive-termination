"""
Sandboxed code execution environment.

Provides `CodeExecutionEnvironment` for running LLM-generated code safely
and `CodeExecutionResult` for capturing outputs and errors.
"""

from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
    CodeExecutionResult,
)

__all__ = ["CodeExecutionEnvironment", "CodeExecutionResult"]
