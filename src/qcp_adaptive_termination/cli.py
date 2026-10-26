"""Command-line entry point for qcp-adaptive-termination.

Delegates to ``scripts.main`` (argument parsing, main), which owns the
script-orchestration layer. Kept as a thin module so the package console entry
point (``qcp_adaptive_termination.cli:main``) stays stable.
"""

from scripts.main import argument_parser, main

__all__ = [
    "argument_parser",
    "main",
]
