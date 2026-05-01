import os
from typing import Literal, Optional
from agents import set_default_openai_client
from dotenv import load_dotenv
import nest_asyncio
import sys
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from modules.prompt import prompt_info_var

load_dotenv()

langfuse = get_client()


class LangfuseProcessor(SpanProcessor):
    def on_start(self, span: "Span", parent_context: Optional[Context] = None) -> None:
        if span.name.startswith(
            "response"
        ):  # The name of the generation spans in your trace
            prompt_info = prompt_info_var.get()
            if prompt_info:
                span.set_attribute("langfuse.prompt.name", prompt_info.get("name"))
                span.set_attribute(
                    "langfuse.prompt.version", prompt_info.get("version")
                )


def initialize(provider: Literal["University", "Work"] = "University"):
    nest_asyncio.apply()
    trace.get_tracer_provider().add_span_processor(LangfuseProcessor())  # type: ignore

    print(f"Provider: {provider}")
    if provider == "Work":
        custom_client = AsyncOpenAI(
            base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY")
        )
        OpenAIAgentsInstrumentor().instrument(exclusive_processor=True)
    else:
        custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)
    set_default_openai_client(custom_client)

    if not langfuse.auth_check():
        print("Langfuse Authentication failed. Please check your credentials and host.")
        sys.exit(1)
