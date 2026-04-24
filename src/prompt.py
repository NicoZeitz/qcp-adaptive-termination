from typing import Any
import chevron


def load_prompt(name: str, version: str | None, parameters: dict[str, Any]) -> str:
    path = (
        f"../prompts/{name}.md"
        if version is None
        else f"../prompts/{name}-{version}.md"
    )

    with open(path) as f:
        return chevron.render(f, parameters)
