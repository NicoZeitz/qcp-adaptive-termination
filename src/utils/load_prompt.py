from pathlib import Path
from typing import Any
from langfuse.model import ChatMessageDict

import chevron

PROMPTS_DIR = Path(__file__).parents[2] / "prompts"

_prompt_cache: dict[str, str] = {
    f.stem: f.read_text() for f in PROMPTS_DIR.glob("*.md")
}


def load_prompt(name: str, parameters: dict[str, Any]) -> list[ChatMessageDict]:
    instructions = chevron.render(_prompt_cache[f"{name}-instructions"], parameters)
    user = chevron.render(_prompt_cache[f"{name}-user"], parameters)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
