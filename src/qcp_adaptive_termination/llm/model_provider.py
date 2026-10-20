import json
import logging
import os
from contextvars import ContextVar

import httpx
from agents import (
    MultiProvider,
    OpenAIProvider,
)
from agents.models.multi_provider import MultiProviderMap
from httpx import AsyncClient
from openai import AsyncOpenAI

_multi_provider: MultiProvider | None = None

# When set (e.g. by the completeness evaluator), each [REASONING] line is tagged
# with this label so reasoning can be associated back to query/step/agents.
llm_reasoning_context: ContextVar[str | None] = ContextVar("llm_reasoning_context", default=None)


def get_multi_provider() -> MultiProvider:
    global _multi_provider
    if _multi_provider is None:
        _multi_provider = _setup_multi_provider()

    return _multi_provider


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


def _extract_reasoning_body(response: httpx.Response) -> list[str]:
    """Pull chat-completion reasoning from a raw provider response.

    Real OpenAI models surface reasoning as ``reasoning_content`` on the message;
    the university vLLM endpoint returns it as plain ``message.reasoning``. The
    AGENTS SDK only reads ``reasoning_content``, so vLLM reasoning is otherwise
    dropped before it ever reaches ``runner.py``.
    """
    if not response.is_success:
        return []
    try:
        payload = response.json()
    except Exception:
        return []

    texts: list[str] = []
    for choice in payload.get("choices", []) or []:
        message = choice.get("message", {}) or {}
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            texts.append(reasoning)
    return texts


async def _modify_response_hook(response: httpx.Response):
    """Log the model's chain-of-thought reasoning to stdout.log.

    The SDK discards vLLM's ``message.reasoning``, so we capture it here at the
    HTTP layer. If ``llm_reasoning_context`` is set (e.g. by the completeness
    evaluator), the label is prepended so reasoning can be associated back to
    query/step/agents.
    """
    # httpx response hooks fire before the body is read; read it here so we can
    # parse reasoning. httpx caches the loaded content, so the SDK's subsequent
    # read/json() reuses these bytes and is unaffected.
    await response.aread()

    reasoning = _extract_reasoning_body(response)
    tag = llm_reasoning_context.get()
    prefix = f"[{tag}] " if tag else ""
    for text in reasoning:
        print(f"[REASONING]{prefix}{text}", flush=True)
    return response


def _add_provider(
    prefix: str,
    provider_map: MultiProviderMap,
    *,
    responses_api: bool = True,
    http_client: AsyncClient | None = None,
):
    base_url = os.getenv(f"{prefix.upper()}_BASE_URL")
    api_key = os.getenv(f"{prefix.upper()}_API_KEY")

    if not base_url:
        logging.warning(f"{prefix.upper()}_BASE_URL not set; skipping provider registration.")
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "not-required",
        http_client=http_client,
        # timeout=httpx.Timeout(connect=60, read=600, write=600, pool=60),
    )
    provider = OpenAIProvider(
        openai_client=client,
        use_responses=responses_api,
    )
    provider_map.add_provider(prefix.lower(), provider)


def _response_hook_client() -> AsyncClient:
    """Shared client that only logs reasoning (applies to every provider).

    The reasoning-logging response hook is generic (safe on any provider —
    it just extracts `message.reasoning`/`reasoning_content` when present).
    The vLLM-specific request hook stays on the UNIVERSITY client only.
    """
    return AsyncClient(event_hooks={"response": [_modify_response_hook]})


def _setup_multi_provider() -> MultiProvider:
    provider_map = MultiProviderMap()
    _add_provider("OPENAI", provider_map=provider_map, http_client=_response_hook_client())
    _add_provider(
        "UNIVERSITY",
        provider_map=provider_map,
        responses_api=False,
        http_client=AsyncClient(
            event_hooks={
                "request": [_modify_request_hook],
                "response": [_modify_response_hook],
            }
        ),
    )
    _add_provider("LOCAL", provider_map=provider_map, http_client=_response_hook_client())
    return MultiProvider(provider_map=provider_map)
