from typing import Any
import chevron
from langfuse import get_client

def load_prompt_old(name: str, version: str | None, parameters: dict[str, Any]) -> str:
    path = (
        f"../prompts/{name}.md"
        if version is None
        else f"../prompts/{name}-{version}.md"
    )

    with open(path) as f:
        return chevron.render(f, parameters)

def load_prompt(name: str, parameters: dict[str, Any]) -> str:
    print("Getting prompt: " + name)
    prompt = get_client().get_prompt(name)
    return prompt.compile(**parameters)