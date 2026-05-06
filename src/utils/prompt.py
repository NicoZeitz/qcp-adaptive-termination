from contextvars import ContextVar
from typing import Any, TypedDict
from langfuse import get_client


class PromptInfo(TypedDict):
    name: str
    version: int


prompt_info_var = ContextVar[PromptInfo | None]("prompt_info", default=None)


def load_prompt(name: str, parameters: dict[str, Any]) -> str:
    prompt = get_client().get_prompt(name)
    # Limitation: currently only supports one prompt at a time
    # https://github.com/orgs/langfuse/discussions/7608
    # Use first prompt if multiple versions exist (e.g. Instruction Prompt)
    if prompt_info_var.get() is None:
        prompt_info_var.set(
            {
                "name": prompt.name,
                "version": prompt.version,
            }
        )

    return prompt.compile(**parameters)
