from pathlib import Path
from typing import Any, Optional
from langfuse.model import ChatMessageDict
from opentelemetry.context import Context
from opentelemetry.trace import Span
from opentelemetry.sdk.trace import SpanProcessor

import chevron

PROMPTS_DIR = Path(__file__).parents[2] / "prompts"


def load_prompt(name: str, parameters: dict[str, Any]) -> list[ChatMessageDict]:
    instructions = chevron.render(
        (PROMPTS_DIR / f"{name}-instructions.md").read_text(), parameters
    )
    user = chevron.render((PROMPTS_DIR / f"{name}-user.md").read_text(), parameters)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]


class LangfuseProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        pass  # prompt version tracking removed; Langfuse used for tracing only
