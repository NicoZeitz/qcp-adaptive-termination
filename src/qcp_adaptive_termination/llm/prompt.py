from collections.abc import Mapping
from importlib.resources import files
from typing import Any, cast

import chevron
from openai.types.responses import EasyInputMessageParam

_PROMPTS = files("qcp_adaptive_termination.prompts")

_prompt_cache: dict[str, str] = {}


def _get_prompt(name: str) -> str:
    if name not in _prompt_cache:
        _prompt_cache[name] = _PROMPTS.joinpath(f"{name}.md").read_text()
    return _prompt_cache[name]


def load_prompt(name: str, parameters: Mapping[str, Any]) -> list[EasyInputMessageParam]:
    parameters = cast(dict[str, Any], parameters)
    instructions = chevron.render(_get_prompt(f"{name}-instructions"), parameters)
    user = chevron.render(_get_prompt(f"{name}-user"), parameters)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
