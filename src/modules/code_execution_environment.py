"""Restricted Python execution environment with a builtin allowlist and isolated I/O."""

import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import io
import os
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable
import uuid

from agents import CustomSpanData, FunctionTool, RunContextWrapper, Span, function_tool

from modules.model import (
    CHAR_LIMIT_LLM,
    CHAR_LIMIT_SPAN,
    AgentContext,
    preview_text,
    serialize_value,
)

# Explicit allowlist: only builtins that cannot access the filesystem, network, or process state.
_SAFE_BUILTINS = {
    key: getattr(builtins, key)
    for key in [
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "callable",
        "chr",
        "delattr",
        "dir",
        "divmod",
        "format",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "aiter",
        "len",
        "max",
        "min",
        "next",
        "anext",
        "oct",
        "ord",
        "pow",
        "print",
        "repr",
        "round",
        "setattr",
        "sorted",
        "sum",
        "vars",
        "None",
        "Ellipsis",
        "NotImplemented",
        "False",
        "True",
        "bool",
        "memoryview",
        "bytearray",
        "bytes",
        "classmethod",
        "complex",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "property",
        "int",
        "list",
        "map",
        "object",
        "range",
        "reversed",
        "set",
        "slice",
        "staticmethod",
        "str",
        "super",
        "tuple",
        "type",
        "zip",
        "BaseException",
        "BaseExceptionGroup",
        "Exception",
        "GeneratorExit",
        "KeyboardInterrupt",
        "SystemExit",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BufferError",
        "EOFError",
        "ImportError",
        "LookupError",
        "MemoryError",
        "NameError",
        "OSError",
        "ReferenceError",
        "RuntimeError",
        "StopAsyncIteration",
        "StopIteration",
        "SyntaxError",
        "SystemError",
        "TypeError",
        "ValueError",
        "Warning",
        "FloatingPointError",
        "OverflowError",
        "ZeroDivisionError",
        "BytesWarning",
        "DeprecationWarning",
        "EncodingWarning",
        "FutureWarning",
        "ImportWarning",
        "PendingDeprecationWarning",
        "ResourceWarning",
        "RuntimeWarning",
        "SyntaxWarning",
        "UnicodeWarning",
        "UserWarning",
        "BlockingIOError",
        "ChildProcessError",
        "ConnectionError",
        "FileExistsError",
        "FileNotFoundError",
        "InterruptedError",
        "IsADirectoryError",
        "NotADirectoryError",
        "PermissionError",
        "ProcessLookupError",
        "TimeoutError",
        "IndentationError",
        "IndexError",
        "KeyError",
        "ModuleNotFoundError",
        "NotImplementedError",
        # "PythonFinalizationError",
        "RecursionError",
        "UnboundLocalError",
        "UnicodeError",
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "TabError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeTranslateError",
        "ExceptionGroup",
        "EnvironmentError",
        "IOError",
    ]
}

# Explicitly set to None
_UNSAFE_BUILTINS = {
    key: None
    for key in [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__build_class__",
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "__debug__",
        "_IncompleteInputError",
        "open",
        "quit",
        "exit",
        "copyright",
        "credits",
        "license",
        "help",
        "_",
    ]
}


@dataclass
class CodeExecutionResult:
    """Holds the outcome of a single code execution, including captured output and surviving locals."""

    code: str
    """The code that was executed"""

    stdout: str
    """Everything written to sys.stdout during execution."""

    stderr: str
    """Everything written to sys.stderr during execution."""

    exception: Exception | None
    """First unhandled exception, or None on clean exit."""

    locals: dict[str, Any]
    """All local variables that survived after execution."""

    formatted: str
    """A short formatted string that can be given to the LLM as information"""


class CodeExecutionEnvironment:
    """Stateful REPL-like environment that persists locals across calls while keeping globals immutable."""

    def __init__(
        self, variables: dict[str, Any] | None, tools: dict[str, Callable] | None
    ):
        """
        Args:
            variables: Pre-seeded local variables available to all executions.
            tools: Callable objects injected into the global namespace.
        """
        self._lock = threading.Lock()
        self.variables = variables or {}
        self.tools = tools or {}
        self.code_executions: list[CodeExecutionResult] = []

        self.initial_globals: dict[str, Any] = {
            **_SAFE_BUILTINS.copy(),
            **_UNSAFE_BUILTINS.copy(),
            **(self.tools.copy()),
            "__name__": "__main__",
        }
        self.initial_locals: dict[str, Any] = {**self.variables.copy()}

        self.globals = self.initial_globals.copy()
        self.locals = self.initial_locals.copy()

    def execute_code(self, code: str) -> CodeExecutionResult:
        """Execute `code` in an isolated working directory and return captured output.

        Any new non-private names defined by the code are merged into `self.locals` so they
        survive into the next call. Exceptions are caught and returned rather than raised.
        """
        with (
            self._capture_output() as (stdout_buf, stderr_buf),
            self._temp_cwd(),
            self._exec_variables() as exec_variables,
        ):
            try:
                exec(code, exec_variables, exec_variables)
                exception = None
            except Exception as e:
                exception = e
            finally:
                for key, value in exec_variables.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value
                locals = self.locals

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()

        execution_result = CodeExecutionResult(
            code=code,
            stdout=stdout,
            stderr=stderr,
            exception=exception,
            locals=locals,
            formatted=self._format_result(stdout, stderr, exception, locals),
        )

        self.code_executions.append(execution_result)
        return execution_result

    def as_function_tool(self) -> FunctionTool:
        @function_tool(
            name_override="execute_code",
            timeout=60,
            timeout_behavior="error_as_result",
        )
        async def execute_python_code(
            ctx: RunContextWrapper[AgentContext], code: str
        ) -> str:
            """
            Execute python code. You will only see partial output results.
            """
            return self.execute_code(code).formatted

        return execute_python_code

    def get_code_executions(self) -> list[CodeExecutionResult]:
        return self.code_executions

    def to_span_data(self) -> list[Any]:
        return [
            {
                "code": preview_text(code_execution.code, CHAR_LIMIT_SPAN),
                "stdout": preview_text(code_execution.stdout, CHAR_LIMIT_SPAN),
                "stderr": preview_text(code_execution.stderr, CHAR_LIMIT_SPAN),
                "exception": serialize_value(code_execution.exception),
                "locals": str(
                    [
                        f"{k}: {type(v).__name__}"
                        for k, v in code_execution.locals.items()
                        if not k.startswith("_")
                    ]
                ),
                # "formatted": code_execution.formatted,
            }
            for code_execution in self.code_executions
        ]


    def _format_result(
        self,
        stdout: str,
        stderr: str,
        exception: Exception | None,
        locals: dict[str, Any],
    ) -> str:
        formatted = ""
        if len(stdout) > 0:
            formatted += f"\nStdout: {preview_text(stdout, CHAR_LIMIT_LLM)}"
        else:
            formatted += "\nNo Stout (if output was expected: you must use print())"

        if len(stderr) > 0:
            formatted += f"\nStderr: {preview_text(stderr, CHAR_LIMIT_LLM)}"

        if exception is not None:
            formatted += f"\n{type(exception).__name__}: {preview_text(str(exception), CHAR_LIMIT_LLM)}"

        available = ",".join(
            [
                f"'{k}': {type(v).__name__}"
                for k, v in locals.items()
                if not k.startswith("_")
            ]
        )

        formatted += f"\nLocals: {available}"
        return formatted.strip()

    @contextmanager
    def _exec_variables(self):
        """Yield a merged globals+locals dict for exec, then restore initial globals afterward.

        Globals are reset to their initial state after each execution so that user code cannot
        permanently overwrite injected tools or safe-builtin entries.
        """
        combined = {**self.globals, **self.locals}
        try:
            yield combined
        finally:
            self.globals = {**self.globals.copy(), **self.initial_globals.copy()}
            self.locals = {**self.locals.copy(), **self.initial_locals.copy()}

    # Adapted from [RLM](https://github.com/alexzhang13/rlm), Original source file: rlm/environments/local_repl.py
    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    # Adapted from [RLM](https://github.com/alexzhang13/rlm), Original source file: rlm/environments/local_repl.py
    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        try:
            os.chdir(temp_dir)
            yield
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        finally:
            os.chdir(old_cwd)
