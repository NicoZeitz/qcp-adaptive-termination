import dataclasses
from typing import Any

from agents import (
    Agent,
    AgentOutputSchema,
    FunctionToolResult,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
)

from modules.cache import Cache
from modules.code_execution_environment import CodeExecutionEnvironment
from modules.model import (
    CHAR_LIMIT_LONG_LLM,
    CHAR_LIMIT_SPAN,
    AgentContext,
    DataAgent,
    Need,
    PartialQueryAnswer,
    Symbol,
    own_custom_span,
    preview_text,
    serialize_value,
)
from modules.prompt import load_prompt


class TextDataAgent(DataAgent):
    def __init__(
        self,
        agent: Agent[AgentContext],
        text: str,
    ):
        super().__init__(agent)
        self.text = text.strip()
        self.initialized = False
        self.summary: str = ""

    async def get_data_and_query_capabilities(self) -> str:
        await self._initialize()
        return self.summary + " | Capabilities: Query Text Content"

    async def get_schema_for_query(self, query: str) -> str | None:
        await self._initialize()

        instructions = load_prompt(
            "text_data_agent_relevance_agent_instructions",
            None,
            {
                "summary": self.summary,
                "preview": preview_text(self.text, CHAR_LIMIT_LONG_LLM),
            },  # TODO: full text as context
        )
        prompt = load_prompt(
            "text_data_agent_relevance_agent_prompt", None, {"query": query}
        )

        relevance_agent = self.agent.clone(
            name=f"{self.agent.name}_Relevance_Agent",
            instructions=instructions,
            output_type=bool,
        )

        async def compute_is_relevant():
            return (await Runner.run(relevance_agent, prompt)).final_output

        is_relevant = await Cache.instance().get_or_compute(
            relevance_agent, prompt, bool, compute_is_relevant
        )

        if not is_relevant:
            return None

        return self.summary

    async def answer_query(
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        await self._initialize()

        with own_custom_span("Text Data Agent - answer_query") as span:
            needs: list[Any] = []
            answer: Any | Symbol = Symbol()

            additional_information = [
                {"information": answered_need.need, "data": answered_need.answer}
                for answered_need in answered_needs
            ]

            def raise_needs(new_needs: list[Any]):
                needs.extend(new_needs)

            def final_answer(new_answer: Any):
                nonlocal answer
                answer = new_answer

            code_environment = CodeExecutionEnvironment(
                variables={
                    "query": query,
                    "text": self.text,
                    "summary": self.summary,
                    "additional_information": additional_information,
                },
                tools={
                    "raise_needs": raise_needs,
                    "final_answer": final_answer,
                },
            )

            def custom_tool_handler(
                _context: RunContextWrapper[AgentContext],
                _tool_results: list[FunctionToolResult],
            ) -> ToolsToFinalOutputResult:
                nonlocal answer, needs
                if len(needs) == 0 and isinstance(answer, Symbol):
                    return ToolsToFinalOutputResult(
                        is_final_output=False, final_output=None
                    )

                answer = "" if isinstance(answer, Symbol) else serialize_value(answer)
                needs = [serialize_value(need) for need in needs]

                return ToolsToFinalOutputResult(
                    is_final_output=True,
                    final_output=PartialQueryAnswer(answer=answer, needs=needs),
                )

            code_agent = self.agent.clone(
                name=f"{self.agent.name}_Code_Agent",
                instructions=load_prompt(
                    "text_data_agent_code_agent_instructions",
                    None,
                    {"summary": self.summary},
                ),
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                model_settings=dataclasses.replace(
                    self.agent.model_settings,
                    parallel_tool_calls=False,
                    tool_choice="required",
                ),
                output_type=AgentOutputSchema(
                    PartialQueryAnswer, strict_json_schema=False
                ),
            )

            code_agent_prompt = load_prompt(
                "text_data_agent_code_agent_prompt", None, {"query": query}
            )

            async def compute_code_answer() -> PartialQueryAnswer:
                return (await Runner.run(code_agent, code_agent_prompt)).final_output

            result = await Cache.instance().get_or_compute(
                code_agent,
                code_agent_prompt,
                PartialQueryAnswer,
                compute_code_answer,
            )

            code_environment.append_data_to_span(span)
            span.span_data.data["partial_answer"] = {
                "answer": preview_text(str(result.answer), CHAR_LIMIT_SPAN),
                "needs": [
                    preview_text(str(need), CHAR_LIMIT_SPAN)
                    for need in result.needs or []
                ],
            }

            return result

    async def _initialize(self):
        if self.initialized:
            return

        summary_agent = self.agent.clone(
            name=f"{self.agent.name}_Summary_Agent",
            instructions=load_prompt(
                "text_data_agent_summary_agent_instructions", None, {}
            ),
            output_type=str,
        )

        prompt = load_prompt(
            "text_data_agent_summary_agent_prompt",
            None,
            {"preview": preview_text(self.text, CHAR_LIMIT_LONG_LLM)},
        )

        async def compute_summary():
            return (await Runner.run(summary_agent, prompt)).final_output

        self.summary = await Cache.instance().get_or_compute(
            summary_agent,
            prompt,
            str,
            compute_summary,
        )
        self.initialized = True
