import json
import logging
import os

import httpx
from agents import (
    MultiProvider,
    OpenAIProvider,
)
from agents.models.multi_provider import MultiProviderMap
from openai import AsyncOpenAI

_multi_provider: MultiProvider | None = None


def set_multi_provider(provider: MultiProvider) -> None:
    global _multi_provider
    _multi_provider = provider


def get_multi_provider() -> MultiProvider:
    if _multi_provider is None:
        raise RuntimeError("MultiProvider not initialized — call initialize() first")
    return _multi_provider


async def modify_request_hook(request: httpx.Request):
    if request.method == "POST" and "application/json" in request.headers.get(
        "content-type", ""
    ):
        body_bytes = await request.aread()
        request_json = json.loads(body_bytes.decode("utf-8"))

        request_json["temperature"] = 0.0

        # thinking_mode needs to be disabled for correct output, otherwise it will return output in reasoning, which breaks openai sdk
        thinking_enabled = (
            "reasoning_effort" in request_json
            and request_json["reasoning_effort"] != "none"
        )

        request_json["chat_template_kwargs"] = {
            "thinking_mode": "enabled" if thinking_enabled else "disabled",
            "thinking": True if thinking_enabled else False,
            "reasoning_effort": request_json.get("reasoning_effort", "none"),
        }
        if "reasoning_effort" in request_json:
            del request_json["reasoning_effort"]

        # if tool choice required, tools not empty and response format exist, remove response format
        if (
            request_json.get("tool_choice") == "required"
            and request_json.get("tools")
            and request_json.get("response_format") is not None
        ):
            del request_json["response_format"]

        modified_bytes = json.dumps(request_json).encode("utf-8")
        request.stream = httpx.ByteStream(modified_bytes)
        request.headers["Content-Length"] = str(len(modified_bytes))

    return request


def add_provider(
    prefix: str, provider_map: MultiProviderMap, *, university_hooks: bool = False
):
    base_url = os.getenv(f"{prefix.upper()}_BASE_URL")
    api_key = os.getenv(f"{prefix.upper()}_API_KEY")

    if not base_url:
        logging.warning(
            f"{prefix.upper()}_BASE_URL not set; skipping provider registration."
        )
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(event_hooks={"request": [modify_request_hook]})
        if university_hooks
        else None,
    )
    provider = OpenAIProvider(
        openai_client=client,
        use_responses=False if university_hooks else None,
    )
    provider_map.add_provider(prefix.lower(), provider)


def setup_providers() -> None:
    provider_map = MultiProviderMap()
    add_provider("OPENAI", provider_map=provider_map)
    add_provider("UNIVERSITY", provider_map=provider_map, university_hooks=True)
    add_provider("WORK", provider_map=provider_map)
    set_multi_provider(MultiProvider(provider_map=provider_map))
