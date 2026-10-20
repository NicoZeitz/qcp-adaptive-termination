import json
import logging
import os
from typing import Any

import httpx
from agents import (
    MultiProvider,
    OpenAIChatCompletionsModel,
    OpenAIProvider,
)
from agents.models.interface import Model, ModelResponse
from agents.models.multi_provider import MultiProviderMap
from httpx import AsyncClient
from openai import AsyncOpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from qcp_adaptive_termination.observability.context import get_config

_multi_provider: MultiProvider | None = None


def get_multi_provider() -> MultiProvider:
    global _multi_provider
    if _multi_provider is None:
        _multi_provider = _setup_multi_provider()

    return _multi_provider


async def _log_request_hook(request: httpx.Request):
    if get_config().log_llm_requests and request.method == "POST":
        body_bytes = await request.aread()
        request.stream = httpx.ByteStream(body_bytes)
        logging.info(
            "LLM REQUEST %s\ncurl -s -X POST '%s' \\\n  -H 'Content-Type: application/json' \\\n  -H 'Authorization: Bearer $UNIVERSITY_API_KEY' \\\n  -d '%s'",
            request.url,
            request.url,
            body_bytes.decode("utf-8").replace("'", "'\\''"),
        )


def _extract_last_json_object(text: str) -> str:
    last_brace = text.rfind("{")
    if last_brace == -1:
        return text
    candidate = text[last_brace:]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return text


def _strip_deepseek_prefix(response: ModelResponse) -> ModelResponse:
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for content in item.content:
            if isinstance(content, ResponseOutputText):
                fixed = _extract_last_json_object(content.text)
                if fixed != content.text:
                    content.text = fixed
    return response


class DeepSeekFixModel(OpenAIChatCompletionsModel):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = await super().get_response(*args, **kwargs)
        return _strip_deepseek_prefix(response)


class DeepSeekFixProvider(OpenAIProvider):
    def get_model(self, model_name: str | None) -> Model:
        base = super().get_model(model_name)
        model = DeepSeekFixModel.__new__(DeepSeekFixModel)
        model.__dict__.update(base.__dict__)
        return model


async def _modify_request_hook(request: httpx.Request):
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        body_bytes = await request.aread()
        request_json = json.loads(body_bytes.decode("utf-8"))

        request_json["temperature"] = 0.0

        # thinking_mode needs to be disabled for correct output,
        # otherwise it will return output in reasoning, which breaks openai sdk
        thinking_enabled = "reasoning_effort" in request_json and request_json["reasoning_effort"] != "none"

        request_json["chat_template_kwargs"] = {
            "thinking_mode": "enabled" if thinking_enabled else "disabled",
            "thinking": bool(thinking_enabled),
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


def _add_provider(
    prefix: str,
    provider_map: MultiProviderMap,
    timeout: float,
    *,
    responses_api: bool = True,
    http_client: AsyncClient | None = None,
    provider_class: type[OpenAIProvider] = OpenAIProvider,
):
    base_url = os.getenv(f"{prefix.upper()}_BASE_URL")
    api_key = os.getenv(f"{prefix.upper()}_API_KEY")

    if not base_url:
        logging.warning(f"{prefix.upper()}_BASE_URL not set; skipping provider registration.")
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "not-required",
        http_client=http_client or AsyncClient(event_hooks={"request": [_log_request_hook]}),
        timeout=timeout,
    )
    provider = provider_class(
        openai_client=client,
        use_responses=responses_api,
    )
    provider_map.add_provider(prefix.lower(), provider)


def _setup_multi_provider() -> MultiProvider:
    provider_map = MultiProviderMap()
    timeout = get_config().llm_timeout
    _add_provider("OPENAI", provider_map=provider_map, timeout=timeout)
    _add_provider(
        "OPENROUTER",
        provider_map=provider_map,
        timeout=timeout,
        responses_api=False,
        http_client=AsyncClient(event_hooks={"request": [_log_request_hook, _modify_request_hook]}),
    )
    _add_provider(
        "UNIVERSITY",
        provider_map=provider_map,
        timeout=timeout,
        responses_api=False,
        http_client=AsyncClient(event_hooks={"request": [_log_request_hook, _modify_request_hook]}),
        provider_class=DeepSeekFixProvider,
    )
    _add_provider("LOCAL", provider_map=provider_map, timeout=timeout)
    return MultiProvider(provider_map=provider_map)
