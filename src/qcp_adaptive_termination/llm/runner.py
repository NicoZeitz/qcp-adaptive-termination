import asyncio
import functools
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    Tool,
    ToolsToFinalOutputFunction,
    TResponseInputItem,
)
from agents.exceptions import ModelBehaviorError
from agents.run_config import DEFAULT_MAX_TURNS
from openai.types.shared import Reasoning
from pydantic import BaseModel, JsonValue

import qcp_adaptive_termination.hooks as hooks
import qcp_adaptive_termination.llm.model_provider as model_provider
import qcp_adaptive_termination.llm.prompt as prompt
from qcp_adaptive_termination.models import AgentRole, CacheMode, RoleConfig
from qcp_adaptive_termination.observability import cache, context, tracing

_AGENT_SEMAPHORE = asyncio.Semaphore(5)


@dataclass(slots=True)
class _SerializedModel:
    """Pickle-safe stand-in for dynamic pydantic outputs that can't be pickled directly."""

    dump: dict[str, Any]

    @staticmethod
    def wrap(input: Any) -> Any | _SerializedModel:
        def _is_dynamic_model(output_type: type[Any]) -> bool:
            module = sys.modules.get(output_type.__module__)
            if module is None:
                return True

            obj: Any = module
            for part in output_type.__qualname__.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return True

            return obj is not output_type

        if isinstance(input, BaseModel) and _is_dynamic_model(type(input)):
            return _SerializedModel(dump=input.model_dump())

        return input

    @staticmethod
    def unwrap(input: Any | _SerializedModel, output_type: type[Any] | None = None) -> Any:
        if isinstance(input, _SerializedModel):
            if output_type is None:
                raise RuntimeError("Output type must be provided for serialized models")

            if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
                raise RuntimeError(f"Expected a BaseModel output for {output_type!r}, got a serialized model")

            return output_type.model_validate(input.dump)
        return input


@dataclass(slots=True)
class AgentRunResult[TResult]:
    """Parsed output plus the raw conversation continuation for the next call."""

    output: TResult
    """Parsed final result."""
    continuation_input: list[TResponseInputItem]
    """Raw conversation items (``to_input_list()``: reasoning, messages, tool calls/outputs)."""


@functools.cache
def _cached_json_schema(output_type: type[BaseModel]) -> dict:
    return output_type.model_json_schema()


def _resolve_model_config(role: AgentRole) -> tuple[str, Reasoning]:
    role_cfg: RoleConfig = context.get_config().agent_configs[role.value]
    reasoning = Reasoning(effort=role_cfg.reasoning_effort, summary=role_cfg.reasoning_summary)
    return role_cfg.model, reasoning


def _get_output_schema(output_type: type | None) -> JsonValue:
    if output_type is None or output_type is str:
        return {"type": "string"}
    if output_type is bool:
        return {"type": "boolean"}
    if output_type is int:
        return {"type": "integer"}
    if output_type is float:
        return {"type": "number"}
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return _cached_json_schema(output_type)
    raise ValueError(f"Unsupported output type: {output_type}")


async def run_agent[TResult](
    role: AgentRole,
    prompt_parameters: Mapping[str, str | int | float | bool | None],
    *,
    agent_name: str | None = None,
    max_turns: int | None = None,
    context: JsonValue = None,
    output_type: type[TResult] | None = str,  # ty: ignore[invalid-parameter-default]
    tools: list[FunctionTool] | None = None,
    reset_tool_choice: bool = True,
    tool_use_behavior: (Literal["run_llm_again", "stop_on_first_tool"] | ToolsToFinalOutputFunction) = "run_llm_again",
    parallel_tool_calls: bool = True,
    tool_choice: Literal["auto", "required", "none"] | None = None,
    cache_mode: CacheMode | None = None,
    previous_input: list[TResponseInputItem] | None = None,
    label: str | None = None,
) -> AgentRunResult[TResult]:
    display_name = f"{agent_name} - {role.value}" if agent_name else role.value
    model, reasoning = _resolve_model_config(role)
    if max_turns is None:
        max_turns = DEFAULT_MAX_TURNS * 2

    instructions, prompts = prompt.load_prompt(role.value, prompt_parameters)
    tools = tools or []
    tool_use = (
        tool_use_behavior
        if isinstance(tool_use_behavior, str)
        else "function" + getattr(tool_use_behavior, "__name__", repr(tool_use_behavior))
    )

    output = _get_output_schema(output_type)

    @cache.cached(mode=cache_mode)
    async def compute_value(
        instructions: str,
        input: list[TResponseInputItem],
        context: JsonValue,
        output: JsonValue,
        function_tools: list[dict[str, Any]],
        reset_tool_choice: bool,
        tool_use: str,
        parallel_tool_calls: bool,
        tool_choice: Literal["auto", "required", "none"] | None,
        label: str | None,
    ) -> AgentRunResult[Any]:
        nonlocal tool_use_behavior
        agent = Agent[JsonValue](
            name=display_name,
            model=model,
            model_settings=ModelSettings(
                reasoning=reasoning,
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=tool_choice,
            ),
            instructions=instructions,
            tools=cast(list[Tool], tools) or [],
            tool_use_behavior=tool_use_behavior,
            reset_tool_choice=reset_tool_choice,
            output_type=output_type,
        )

        tracing.indented_logging(f"Running agent: {display_name}")

        try:
            async with _AGENT_SEMAPHORE:
                result = await Runner.run(
                    agent,
                    input,
                    context=context,
                    max_turns=max_turns,
                    run_config=RunConfig(model_provider=model_provider.get_multi_provider()),
                )
        except ModelBehaviorError as e:
            tracing.indented_logging(f"Model behavior error: {e}")
            raise e

        final_output = _SerializedModel.wrap(cast(TResult, result.final_output))
        return AgentRunResult(
            output=final_output,
            continuation_input=result.to_input_list(),
        )

    cached = await compute_value(
        instructions,
        input=[*(previous_input or []), *prompts],
        context=context,
        output=output,
        function_tools=list(
            map(
                lambda t: {
                    "name": t.name,
                    "description": t.description,
                    "params_json_schema": t.params_json_schema,
                    "strict_json_schema": t.strict_json_schema,
                },
                tools,
            )
        ),
        reset_tool_choice=reset_tool_choice,
        tool_use=tool_use,
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        label=label,
    )
    output = _SerializedModel.unwrap(cached.output, output_type)
    continuation_input = cached.continuation_input

    # Fire on both real runs and cache hits; the raw continuation is passed so
    # consumers can render a transcript / extract reasoning from it directly.
    await hooks.fire_post(
        hooks.HookEvent.LLM_RUN_COMPLETED,
        role=role,
        agent_name=display_name,
        label=label,
        continuation_input=continuation_input,
        output=output,
    )

    return AgentRunResult(
        output=output,
        continuation_input=continuation_input,
    )


async def run_llm[TResult](
    role: AgentRole,
    prompt_parameters: Mapping[str, str | int | float | bool | None],
    *,
    agent_name: str | None = None,
    output_type: type[TResult] | None = str,  # ty: ignore[invalid-parameter-default]
    context: JsonValue = None,
    cache_mode: CacheMode | None = None,
) -> TResult:
    result = await run_agent(
        role=role,
        agent_name=agent_name,
        prompt_parameters=prompt_parameters,
        max_turns=1,
        output_type=output_type,
        context=context,
        tools=[],
        cache_mode=cache_mode,
    )
    return result.output
