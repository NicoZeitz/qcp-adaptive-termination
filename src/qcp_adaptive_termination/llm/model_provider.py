"""Provide model providers (OpenAI and university vLLM) for the agent runner."""

import json
import logging
import os
from typing import Any

import httpx
from agents import (
    MultiProvider,
    OpenAIProvider,
)
from agents.models.multi_provider import MultiProviderMap
from httpx import AsyncBaseTransport, AsyncClient, Request, Response
from openai import AsyncOpenAI

from qcp_adaptive_termination.observability import context

_multi_provider: MultiProvider | None = None


def get_multi_provider() -> MultiProvider:
    """Return the lazily-initialized MultiProvider."""
    global _multi_provider
    if _multi_provider is None:
        _multi_provider = _setup_multi_provider()

    return _multi_provider


class _vLlmTransport(AsyncBaseTransport):
    """Transport for the university vLLM chat-completions endpoint."""

    def __init__(self, **kwargs):
        """Initialize the vLLM transport."""
        self._inner = httpx.AsyncHTTPTransport(**kwargs)

    async def handle_async_request(self, request: Request) -> Response:
        """Modify and forward a request, adapting its response."""
        await self._modify_request(request)

        response = await self._inner.handle_async_request(request)

        if not response.is_success:
            return response

        if "application/json" not in response.headers.get("content-type", ""):
            return response

        await response.aread()

        content = response.json()
        content = self._modify_response_content(content)

        return Response(
            status_code=response.status_code,
            headers=response.headers,
            content=content,
            request=request,
            extensions=response.extensions,
            history=response.history,
        )

    async def aclose(self):
        """Close the inner transport."""
        await self._inner.aclose()

    @staticmethod
    async def _modify_request(request: Request):
        """Adapt a request to the university vLLM chat-completions dialect."""

        body_bytes = await request.aread()
        request_json = json.loads(body_bytes.decode("utf-8"))

        # Mirror fields `reasoning` and `reasoning_content` in the request messages
        for message in request_json.get("messages", []) or []:
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning

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

    @staticmethod
    def _modify_response_content(payload: dict[str, Any]) -> bytes:
        """Mirror ``reasoning`` and ``reasoning_content`` on the response."""

        for choice in payload.get("choices", []) or []:
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning")

            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning

        return json.dumps(payload).encode("utf-8")


def _add_provider(
    prefix: str,
    provider_map: MultiProviderMap,
    *,
    responses_api: bool = True,
):
    """Register a provider from environment variables into the provider map."""
    base_url = os.getenv(f"{prefix.upper()}_BASE_URL")
    api_key = os.getenv(f"{prefix.upper()}_API_KEY")

    if not base_url:
        logging.warning(f"{prefix.upper()}_BASE_URL not set; skipping provider registration.")
        return

    cfg = context.get_config().execution
    http_client = AsyncClient(transport=_vLlmTransport()) if prefix.upper() == "UNIVERSITY" else AsyncClient()
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "not-required",
        http_client=http_client,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
    )
    provider = OpenAIProvider(
        openai_client=client,
        use_responses=responses_api,
    )
    provider_map.add_provider(prefix.lower(), provider)


def _setup_multi_provider() -> MultiProvider:
    """Set up and return the MultiProvider from configured providers."""
    provider_map = MultiProviderMap()
    _add_provider("OPENAI", provider_map=provider_map)
    _add_provider("OPENROUTER", provider_map=provider_map)
    _add_provider(
        "UNIVERSITY",
        provider_map=provider_map,
        responses_api=False,
    )
    return MultiProvider(provider_map=provider_map)
