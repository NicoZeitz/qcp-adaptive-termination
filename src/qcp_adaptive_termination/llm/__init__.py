"""
LLM provider management and prompt loading.

Manages a multi-provider LLM backend (setup, get, set) and exposes
`load_prompt` for loading prompt templates from disk.
"""

from qcp_adaptive_termination.llm.providers import (
    get_multi_provider,
    set_multi_provider,
    setup_providers,
)
from qcp_adaptive_termination.llm.prompt import load_prompt

__all__ = [
    "get_multi_provider",
    "set_multi_provider",
    "setup_providers",
    "load_prompt",
]
