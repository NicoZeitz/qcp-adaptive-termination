from typing import Literal, Type, cast

from agents import (
    Agent,
    AgentOutputSchemaBase,
    ModelSettings,
    RunResult,
    Runner,
    StopAtTools,
    TResponseInputItem,
    Tool,
    ToolsToFinalOutputFunction,
)
from agents.model_settings import ToolChoice

from qtypes.agent_configuration import AgentConfiguration
from utils.prompt import load_prompt


async def run_agent[TResult](
    name_postfix: str,
    agent_configuration: AgentConfiguration,
    prompt_name: str,
    prompt_parameters: dict[str, str],
    *,
    context: object | None = None,
    output_type: Type[TResult] | AgentOutputSchemaBase | None = str,
    tools: list[Tool] | None = None,
    reset_tool_choice: bool = True,
    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"]
        | StopAtTools
        | ToolsToFinalOutputFunction
    ) = "run_llm_again",
    parallel_tool_calls: bool | None = True,
    tool_choice: ToolChoice | None = None,
) -> tuple[TResult, RunResult]:
    tools = tools or []

    prompts = load_prompt(prompt_name, prompt_parameters)
    # use 1st message as instructions, rest as part of prompt
    instructions, prompts = prompts[0], prompts[1:]

    agent = Agent[object](
        name=f"{agent_configuration.name} - {name_postfix}",
        model=agent_configuration.model,
        model_settings=ModelSettings(
            reasoning=agent_configuration.reasoning,
            verbosity=agent_configuration.verbosity,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        ),
        instructions=instructions["content"],
        tools=tools,
        tool_use_behavior=tool_use_behavior,
        reset_tool_choice=reset_tool_choice,
        output_type=output_type,
    )

    result = await Runner.run(agent, cast(list[TResponseInputItem], prompts), context=context)
    return cast(TResult, result.final_output), result
