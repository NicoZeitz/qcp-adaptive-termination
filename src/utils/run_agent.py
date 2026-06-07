from typing import Callable, Literal, Type, TypeVar, cast
import re

from agents import (
    Agent,
    AgentOutputSchemaBase,
    FunctionTool,
    ModelSettings,
    Runner,
    TResponseInputItem,
    Tool,
    ToolsToFinalOutputFunction,
)
from agents.agent_output import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
from agents.run_config import DEFAULT_MAX_TURNS
from pydantic import BaseModel, JsonValue

from qtypes.agent_configuration import AgentConfiguration
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from utils.cache import CacheMode, cached
from utils.load_prompt import load_prompt
from utils.span import indented_logging

TResult = TypeVar("TResult", bound=JsonValue | PartialQueryAnswer)


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    candidate = re.sub(r"^```json", "", stripped)
    candidate = re.sub(r"```$", "", candidate)
    return candidate.strip()


class MarkdownTolerantOutputSchema(AgentOutputSchemaBase):
    """Delegates to AgentOutputSchema but retries with stripped markdown fences."""

    def __init__(self, output_type: type[BaseModel], strict_json_schema: bool = True):
        self._delegate = AgentOutputSchema(
            output_type=output_type,
            strict_json_schema=strict_json_schema,
        )

    def is_plain_text(self) -> bool:
        return self._delegate.is_plain_text()

    def name(self) -> str:
        return self._delegate.name()

    def json_schema(self) -> dict[str, object]:
        return self._delegate.json_schema()

    def is_strict_json_schema(self) -> bool:
        return self._delegate.is_strict_json_schema()

    def validate_json(self, json_str: str) -> object:
        try:
            return self._delegate.validate_json(json_str)
        except ModelBehaviorError:
            cleaned = _strip_markdown_code_fences(json_str)
            if cleaned == json_str:
                raise
            return self._delegate.validate_json(cleaned)


async def run_agent[TResult](
    agent_configuration: AgentConfiguration,
    prompt_name: str,
    prompt_parameters: dict[str, str],
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
    get_actual_result: Callable[[], TResult] | None = None,
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
    agent_output_type: Type[TResult] | None | AgentOutputSchemaBase = output_type

    if output_type is None or output_type is str:
        output = {"type": "string"}
    elif output_type is bool:
        output = {"type": "boolean"}
    elif output_type is int:
        output = {"type": "integer"}
    elif output_type is float:
        output = {"type": "number"}
    elif isinstance(output_type, type) and (
        issubclass(output_type, PartialQueryAnswer)
        or issubclass(output_type, PartialQueryAnswerDummy)
    ):
        output = {"type": "PartialQueryAnswer"}
        deserializable_classes.append(PartialQueryAnswer)
        agent_output_type = MarkdownTolerantOutputSchema(output_type)
    elif isinstance(output_type, type) and issubclass(output_type, BaseModel):
        output = output_type.model_json_schema()
        deserializable_classes.append(output_type)
        agent_output_type = MarkdownTolerantOutputSchema(output_type)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    @cached(
        name_prefix=agent_name,
        mode=cache_mode,
        deserializable_classes=deserializable_classes,
    )
    async def compute_value(
        agent_name: str,
        agent_configuration: AgentConfiguration,
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
            output_type=agent_output_type,
        )

        indented_logging(f"Running agent: {agent_name}")
        result = await Runner.run(
            agent,
            cast(list[TResponseInputItem], prompts),
            context=context,
            max_turns=agent_configuration.max_turns or DEFAULT_MAX_TURNS,
        )
        if get_actual_result is None:
            return result.final_output

        return get_actual_result()

    return await compute_value(
        agent_name,
        agent_configuration,
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
