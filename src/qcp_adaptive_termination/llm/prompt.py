from collections.abc import Mapping
from importlib.resources import files
from typing import Any, TypedDict, cast

import chevron

_PROMPTS = files("qcp_adaptive_termination.prompts")

_prompt_cache: dict[str, str] = {}


class PromptMessage(TypedDict):
    role: str
    content: str


def _get_prompt(name: str) -> str:
    if name not in _prompt_cache:
        _prompt_cache[name] = _PROMPTS.joinpath(f"{name}.md").read_text()
    return _prompt_cache[name]


def load_prompt(name: str, parameters: Mapping[str, Any]) -> list[PromptMessage]:
    parameters = cast(dict[str, Any], parameters)
    instructions = chevron.render(_get_prompt(f"{name}-instructions"), parameters)
    user = chevron.render(_get_prompt(f"{name}-user"), parameters)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
