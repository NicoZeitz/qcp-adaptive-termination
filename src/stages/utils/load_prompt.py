from contextvars import ContextVar
from typing import Any, Optional, Sequence, TypedDict, cast
from langfuse import get_client
from langfuse.model import ChatMessageDict
from opentelemetry.context import Context
from opentelemetry.trace import Span
from opentelemetry.sdk.trace import SpanProcessor


class PromptInfo(TypedDict):
    name: str
    version: int


prompt_info_var = ContextVar[PromptInfo | None]("prompt_info", default=None)


def load_prompt(name: str, parameters: dict[str, Any]) -> Sequence[ChatMessageDict]:
    prompt = get_client().get_prompt(name, type="chat")
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
    return cast(Sequence[ChatMessageDict], prompt.compile(**parameters))


class LangfuseProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        if span.name.startswith(
            "response"
        ):  # The name of the generation spans in your trace
            prompt_info = prompt_info_var.get()
            if prompt_info:
                span.set_attribute("langfuse.prompt.name", prompt_info.get("name"))
                span.set_attribute(
                    "langfuse.prompt.version", prompt_info.get("version")
                )
