from typing import Awaitable, Callable, Literal, Type, cast

from agents import (
    Agent,
    AgentOutputSchemaBase,
    ModelSettings,
    RunContextWrapper,
    RunState,
    Runner,
    StopAtTools,
    TResponseInputItem,
    Tool,
    ToolsToFinalOutputFunction,
)
from agents.model_settings import ToolChoice

from qtypes.agent_configuration import AgentConfiguration


async def run_agent[TResult, TContext](
    name_postfix: str,
    agent_configuration: AgentConfiguration,
    prompt: str | list[TResponseInputItem] | RunState[TContext],
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            str | Awaitable[str],
        ]
        | None
    ) = None,
    *,
    context: TContext | None = None,
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
) -> TResult:
    tools = tools or []

    agent = Agent[TContext](
        name=f"{agent_configuration.name} - {name_postfix}",
        model=agent_configuration.model,
        model_settings=ModelSettings(
            reasoning=agent_configuration.reasoning,
            verbosity=agent_configuration.verbosity,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        ),
        instructions=instructions,
        tools=tools,
        tool_use_behavior=tool_use_behavior,
        reset_tool_choice=reset_tool_choice,
        output_type=output_type,
    )

    return cast(
        TResult, (await Runner.run(agent, prompt, context=context)).final_output
    )
