from typing import Optional
from dotenv import load_dotenv
import nest_asyncio
import sys
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor

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


def initialize():
    nest_asyncio.apply()
    trace.get_tracer_provider().add_span_processor(LangfuseProcessor())  # type: ignore
    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)

    if not langfuse.auth_check():
        print("Langfuse Authentication failed. Please check your credentials and host.")
        sys.exit(1)

    # custom_client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
    # custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # set_default_openai_client(custom_client)
