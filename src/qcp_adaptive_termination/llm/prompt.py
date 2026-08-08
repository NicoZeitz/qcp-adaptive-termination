from pathlib import Path
from typing import Any, Mapping, cast

import chevron
from langfuse.model import ChatMessageDict

PROMPTS_DIR = Path(__file__).parents[3] / "prompts"

_prompt_cache: dict[str, str] = {}


def _get_prompt(name: str) -> str:
    if name not in _prompt_cache:
        path = PROMPTS_DIR / f"{name}.md"
        _prompt_cache[name] = path.read_text()
    return _prompt_cache[name]


def load_prompt(name: str, parameters: Mapping[str, Any]) -> list[ChatMessageDict]:
    parameters = cast(dict[str, Any], parameters)
    instructions = chevron.render(_get_prompt(f"{name}-instructions"), parameters)
    user = chevron.render(_get_prompt(f"{name}-user"), parameters)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
