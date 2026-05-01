from contextvars import ContextVar
from typing import Any, TypedDict
from langfuse import get_client

class PromptInfo(TypedDict):
    name: str
    version: int

prompt_info_var = ContextVar[PromptInfo | None]("prompt_info", default=None)

def load_prompt(name: str, parameters: dict[str, Any]) -> str:
    prompt = get_client().get_prompt(name)
    prompt_info_var.set({
        "name": prompt.name,
        "version": prompt.version,
    })
    return prompt.compile(**parameters)