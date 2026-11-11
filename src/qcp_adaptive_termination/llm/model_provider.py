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


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _find_first_valid_json(text: str) -> str | None:
    """Return the substring starting at the first '{' that parses as complete JSON, or None."""
    pos = 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            return None
        try:
            json.loads(text[idx:])
            return text[idx:]
        except json.JSONDecodeError:
            pos = idx + 1


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
        """Mirror ``reasoning`` and ``reasoning_content`` on the response and strip DeepSeek JSON prefixes."""

        for choice in payload.get("choices", []) or []:
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning")

            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning

            content = message.get("content")
            if isinstance(content, str) and not _is_valid_json(content):
                fixed = _find_first_valid_json(content)
                if fixed is not None:
                    message["content"] = fixed

        return json.dumps(payload).encode("utf-8")


class _OpenRouterTransport(AsyncBaseTransport):
    """Transport for the OpenRouter chat-completions endpoint.

    Translates the agents-SDK ``reasoning_effort`` field into OpenRouter's
    ``reasoning.enabled`` / ``reasoning.effort`` format and injects optional
    provider routing from the OPENROUTER_PROVIDER_ONLY env var.
    """

    def __init__(self, **kwargs):
        self._inner = httpx.AsyncHTTPTransport(**kwargs)

    async def handle_async_request(self, request: Request) -> Response:
        await self._modify_request(request)
        response = await self._inner.handle_async_request(request)

        if response.is_success and "application/json" in response.headers.get("content-type", ""):
            await response.aread()
            content = self._modify_response_content(response.json())
            return Response(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                request=request,
                extensions=response.extensions,
                history=response.history,
            )

        return response

    async def aclose(self):
        await self._inner.aclose()

    @staticmethod
    def _modify_response_content(payload: dict[str, Any]) -> bytes:
        """Mirror ``reasoning_content`` and ``reasoning`` on the response choices."""
        for choice in payload.get("choices", []) or []:
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    async def _modify_request(request: Request):
        body_bytes = await request.aread()
        body = json.loads(body_bytes.decode("utf-8"))

        reasoning_effort = body.pop("reasoning_effort", None)

        extra_body = body.get("extra_body") or {}

        if reasoning_effort and reasoning_effort != "none":
            extra_body.setdefault("reasoning", {})["enabled"] = True

        provider_only_env = os.getenv("OPENROUTER_PROVIDER_ONLY")
        allow_fallbacks = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "false").lower() != "false"
        if provider_only_env:
            providers = [p.strip() for p in provider_only_env.split(",") if p.strip()]
            extra_body["provider"] = {"only": providers, "allow_fallbacks": allow_fallbacks}

        if extra_body:
            body["extra_body"] = extra_body

        # Mirror reasoning_content for multi-turn continuations
        for message in body.get("messages", []) or []:
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning

        modified = json.dumps(body).encode("utf-8")
        request.stream = httpx.ByteStream(modified)
        request.headers["Content-Length"] = str(len(modified))


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
    if prefix.upper() == "UNIVERSITY":
        http_client = AsyncClient(transport=_vLlmTransport())
    elif prefix.upper() == "OPENROUTER":
        http_client = AsyncClient(transport=_OpenRouterTransport())
    else:
        http_client = AsyncClient()
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
    _add_provider("OPENROUTER", provider_map=provider_map, responses_api=False)
    _add_provider(
        "UNIVERSITY",
        provider_map=provider_map,
        responses_api=False,
    )
    return MultiProvider(provider_map=provider_map)
