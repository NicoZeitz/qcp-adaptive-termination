"""Load and render prompt templates for agent roles."""

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, cast

import chevron
from agents.items import TResponseInputItem

_PROMPTS = files("qcp_adaptive_termination.prompts")

_prompt_cache: dict[str, str] = {}


def _get_prompt(name: str) -> str:
    """Return the cached raw prompt template for a name."""
    if name not in _prompt_cache:
        _prompt_cache[name] = _PROMPTS.joinpath(f"{name}.md").read_text()
    return _prompt_cache[name]


def load_prompt(name: str, parameters: Mapping[str, Any]) -> tuple[str, list[TResponseInputItem]]:
    """Render the instructions and user prompts from a role's templates."""
    parameters = cast(dict[str, Any], parameters)
    instructions = chevron.render(_get_prompt(f"{name}-instructions"), parameters)
    user = chevron.render(_get_prompt(f"{name}-user"), parameters)
    return (
        instructions,
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            }
        ],
    )
