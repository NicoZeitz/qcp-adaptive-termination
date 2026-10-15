import functools
import time
from collections.abc import Mapping
from typing import Literal, cast

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

import qcp_adaptive_termination.llm.model_provider as model_provider
import qcp_adaptive_termination.llm.prompt as prompt
from qcp_adaptive_termination.models import AgentRole, CacheMode, RoleConfig
from qcp_adaptive_termination.observability import cache, context, tracing

@functools.cache
def _cached_json_schema(output_type: type[BaseModel]) -> dict:
    return output_type.model_json_schema()


def _resolve_model_config(role: AgentRole) -> tuple[str, Reasoning]:
    role_cfg: RoleConfig = context.get_config().agent_configs[role.value]
    reasoning = Reasoning(effort=role_cfg.reasoning_effort, summary=role_cfg.reasoning_summary)
    return role_cfg.model, reasoning


def _get_output_and_deserializable_classes(
    output_type: type | None,
) -> tuple[JsonValue, list[type[BaseModel]]]:
    output: JsonValue = None
    deserializable_classes: list[type[BaseModel]] = []

    if output_type is None or output_type is str:
        output = {"type": "string"}
    elif output_type is bool:
        output = {"type": "boolean"}
    elif output_type is int:
        output = {"type": "integer"}
    elif output_type is float:
        output = {"type": "number"}
    elif isinstance(output_type, type) and issubclass(output_type, BaseModel):
        output = _cached_json_schema(output_type)
        deserializable_classes.append(output_type)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    return output, deserializable_classes


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
) -> TResult:
    display_name = f"{agent_name} - {role.value}" if agent_name else role.value
    model, reasoning = _resolve_model_config(role)
    if max_turns is None:
        max_turns = DEFAULT_MAX_TURNS * 2

    prompts = prompt.load_prompt(role.prompt_name, prompt_parameters)
    # use 1st message as instructions, rest as part of prompt
    instructions, prompts = prompts[0], prompts[1:]

    tools = tools or []
    tool_use = (
        tool_use_behavior
        if isinstance(tool_use_behavior, str)
        else "function" + getattr(tool_use_behavior, "__name__", repr(tool_use_behavior))
    )

    output, deserializable_classes = _get_output_and_deserializable_classes(output_type)

    @cache.cached(
        name_prefix=role.value,
        mode=cache_mode,
        deserializable_classes=deserializable_classes,
    )
    async def compute_value(
        instructions: str,
        prompts: list[dict[str, str]],
        context: JsonValue,
        output: JsonValue,
        function_tools: list[dict[str, str]],
        reset_tool_choice: bool,
        tool_use: str,
        parallel_tool_calls: bool,
        tool_choice: Literal["auto", "required", "none"] | None,
    ):
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

        t0 = time.perf_counter()
        try:
            result = await Runner.run(
                agent,
                cast(list[TResponseInputItem], prompts),
                context=context,
                max_turns=max_turns,
                run_config=RunConfig(model_provider=model_provider.get_multi_provider()),
            )
        except ModelBehaviorError as e:
            tracing.indented_logging(f"Model behavior error: {e}")
            raise e
        duration_ms = round((time.perf_counter() - t0) * 1000)

        usage = result.context_wrapper.usage
        tracing.append_to_span_output({
            f"llm.{display_name}.input_tokens": usage.input_tokens,
            f"llm.{display_name}.output_tokens": usage.output_tokens,
            f"llm.{display_name}.duration_ms": duration_ms,
        })

        acc = context.get_usage_accumulator()
        acc.input_tokens += usage.input_tokens
        acc.output_tokens += usage.output_tokens
        acc.llm_calls += 1
        acc.duration_ms += duration_ms

        return result.final_output

    return await compute_value(
        instructions["content"],
        prompts=list(
            map(lambda p: {"role": p["role"], "content": p["content"]}, prompts),
        ),
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
    return await run_agent(
        role=role,
        agent_name=agent_name,
        prompt_parameters=prompt_parameters,
        max_turns=1,
        output_type=output_type,
        context=context,
        tools=[],
        cache_mode=cache_mode,
    )
