from typing import Literal, Type, cast

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    TResponseInputItem,
    Tool,
    ToolsToFinalOutputFunction,
)
from agents.exceptions import ModelBehaviorError
from agents.run_config import DEFAULT_MAX_TURNS
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, JsonValue

from qtypes.agent_configuration import AgentConfiguration
from stages.common.agent_configuration import get_multi_provider
from utils.cache import CacheMode, cached
from utils.load_prompt import load_prompt
from utils.span import indented_logging


async def run_agent[TResult](
    agent_configuration: AgentConfiguration,
    prompt_name: str,
    prompt_parameters: dict[str, str | int | float | bool | None],
    *,
    context: JsonValue = None,
    output_type: Type[TResult] | None = str,
    tools: list[FunctionTool] | None = None,
    reset_tool_choice: bool = True,
    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"] | ToolsToFinalOutputFunction
    ) = "run_llm_again",
    parallel_tool_calls: bool = True,
    tool_choice: Literal["auto", "required", "none"] | None = None,
    cache_mode: CacheMode | None = None,
) -> TResult:
    tools = tools or []

    prompts = load_prompt(prompt_name, prompt_parameters)
    # use 1st message as instructions, rest as part of prompt
    instructions, prompts = prompts[0], prompts[1:]
    agent_name = agent_configuration.name
    tool_use = (
        tool_use_behavior
        if isinstance(tool_use_behavior, str)
        else "function" + tool_use_behavior.__name__
    )

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
        output = output_type.model_json_schema()
        deserializable_classes.append(output_type)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    @cached(
        name_prefix=agent_name,
        mode=cache_mode,
        deserializable_classes=deserializable_classes,
    )
    async def compute_value(
        agent_name: str,
        # agent_configuration: AgentConfiguration,
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
            name=agent_name,
            model=agent_configuration.model,
            model_settings=ModelSettings(
                reasoning=agent_configuration.reasoning,
                verbosity=agent_configuration.verbosity,
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=tool_choice,
            ),
            instructions=instructions,
            tools=cast(list[Tool], tools) or [],
            tool_use_behavior=tool_use_behavior,
            reset_tool_choice=reset_tool_choice,
            output_type=output_type,
        )

        indented_logging(f"Running agent: {agent_name}")
        available_tools = [t.name for t in tools] if tools else []

        try:
            result = await Runner.run(
                agent,
                cast(list[TResponseInputItem], prompts),
                context=context,
                max_turns=agent_configuration.max_turns or DEFAULT_MAX_TURNS,
                run_config=RunConfig(model_provider=get_multi_provider()),
            )
        except ModelBehaviorError as e:
            if e.run_data is None:
                raise
            last_response = e.run_data.raw_responses[-1]
            history = [item.to_input_item() for item in e.run_data.new_items]
            error_outputs: list[TResponseInputItem] = [
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": (
                        f"Error: tool '{item.name}' does not exist. "
                        f"Available tools: {available_tools}. "
                        f"Use only those."
                    ),
                }
                for item in last_response.output
                if isinstance(item, ResponseFunctionToolCall)
                and item.name not in available_tools
            ]
            result = await Runner.run(
                agent,
                cast(list[TResponseInputItem], prompts) + history + error_outputs,
                context=context,
                max_turns=agent_configuration.max_turns or DEFAULT_MAX_TURNS,
                run_config=RunConfig(model_provider=get_multi_provider()),
            )
        return result.final_output

    return await compute_value(
        agent_name,
        # agent_configuration,
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
    agent_configuration: AgentConfiguration,
    prompt_name: str,
    prompt_parameters: dict[str, str | int | float | bool | None],
    *,
    output_type: Type[TResult] | None = str,
    context: JsonValue = None,
    cache_mode: CacheMode | None = None,
) -> TResult:
    single_turn_config = AgentConfiguration(
        name=agent_configuration.name,
        model=agent_configuration.model,
        reasoning=agent_configuration.reasoning,
        max_turns=1,
    )
    return await run_agent(
        agent_configuration=single_turn_config,
        prompt_name=prompt_name,
        prompt_parameters=prompt_parameters,
        output_type=output_type,
        context=context,
        tools=[],
        cache_mode=cache_mode,
    )
