import asyncio
import dataclasses
import json
from typing import Any

from agents import (
    Agent,
    AgentOutputSchema,
    FunctionToolResult,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    custom_span,
)
from pydantic import JsonValue, create_model, Field

from modules.cache import Cache
from modules.code_execution_environment import CodeExecutionEnvironment
from modules.model import (
    AgentContext,
    DataAgent,
    Need,
    NeedNotMet,
    PartialQueryAnswer,
    QueryPlan,
    ReplanningNeeded,
    ReplanningNeededException,
    ResponsibleAgentNotFound,
    Symbol,
    append_json_preview_to_span,
    serialize_value,
)
from modules.prompt import load_prompt


class QueryCoordinator:
    def __init__(self, agent: Agent[AgentContext]):
        self.data_agents: list[DataAgent] = []
        self.agent: Agent[AgentContext] = agent

    def register_data_agent(self, data_agent: DataAgent):
        print("Register data agent:", data_agent.name())
        self.data_agents.append(data_agent)

    def unregister_data_agent(self, data_agent: DataAgent):
        print("Unregister data agent:", data_agent.name())
        self.data_agents.remove(data_agent)

    async def collaborative_query_answering(self, query: str) -> JsonValue:
        agent_registry = await self._build_agent_registry()

        # 1. Query Coordinator identifies Relevant Agents Based on Capabilities.
        relevant_agents = await self._filter_relevant_agents(query, agent_registry)

        # 2. Data Agents are Asked if they participate
        participating_agents = await self._ask_data_agents_for_participation(
            relevant_agents, query
        )

        # 3. Coordinator Drafts Query Plan & Initializes Stack Based on Previously Identified Agents.
        query_plan = await self._draft_query_plan(query, participating_agents)

        result = await self._execute_query_plan(query, query_plan, participating_agents)
        return result

    async def _build_agent_registry(self) -> str:
        with custom_span("Build Agent Registry") as span:
            capabilities = await asyncio.gather(
                *(agent.get_data_and_query_capabilities() for agent in self.data_agents)
            )

            agent_registry = ""
            for agent, capability in zip(self.data_agents, capabilities):
                span.span_data.data[f"{agent.name()}_capability"] = capability
                agent_registry += f"{agent.name()}: {capability}\n"

            span.span_data.data["agent_registry"] = agent_registry

            return agent_registry.strip()

    async def _filter_relevant_agents(
        self, query: str, agent_registry: str
    ) -> list[DataAgent]:
        with custom_span("Filtering Relevant Agents") as span:
            agent_fields: dict[str, Any] = {
                agent.name(): (bool, Field(...)) for agent in self.data_agents
            }
            RelevanceModel = create_model("RelevanceModel", **agent_fields)

            instructions = load_prompt(
                "query_coordinator_relevance_agent_instructions",
                None,
                {"agent_registry": agent_registry},
            )
            prompt = load_prompt(
                "query_coordinator_relevance_agent_prompt", None, {"query": query}
            )

            relevance_agent = self.agent.clone(
                name=f"{self.agent.name}_Relevance_Agent",
                instructions=instructions,
                output_type=RelevanceModel,
            )

            async def compute_relevances():
                return (await Runner.run(relevance_agent, prompt)).final_output

            relevances = await Cache.instance().get_or_compute(
                relevance_agent,
                prompt,
                RelevanceModel,
                compute_relevances,
            )

            relevant_agents = []
            for agent in self.data_agents:
                relevant = getattr(relevances, agent.name(), False)

                span.span_data.data[f"{agent.name()}_relevant"] = relevant
                if relevant:
                    relevant_agents.append(agent)

            return relevant_agents

    async def _ask_data_agents_for_participation(
        self, relevant_agents: list[DataAgent], query: str
    ) -> list[tuple[DataAgent, str]]:
        with custom_span("Asking Data Agents for Participation") as span:
            schemas = await asyncio.gather(
                *(agent.get_schema_for_query(query) for agent in relevant_agents)
            )

            participating_agents = []
            for agent, schema in zip(relevant_agents, schemas):
                relevant = schema is not None

                span.span_data.data[f"{agent.name()}_relevant"] = relevant
                if relevant:
                    participating_agents.append((agent, schema))

            return participating_agents

    async def _draft_query_plan(
        self, query: str, participating_agents: list[tuple[DataAgent, str]]
    ) -> QueryPlan:
        with custom_span("Draft Query Plan") as span:
            QueryPlanModel = QueryPlan.create_query_plan_model(participating_agents)
            participating_agents_str = "\n".join(
                [
                    f"【{agent.name()}】\n {capability}"
                    for agent, capability in participating_agents
                ]
            )

            instructions = load_prompt(
                "query_coordinator_query_plan_agent_instructions",
                None,
                {"participating_agents": participating_agents_str},
            )
            prompt = load_prompt(
                "query_coordinator_query_plan_agent_prompt", None, {"query": query}
            )

            query_plan_agent = self.agent.clone(
                name=f"{self.agent.name}_Query_Plan_Agent",
                instructions=instructions,
                output_type=QueryPlanModel,
            )

            async def compute_query_plan():
                return (await Runner.run(query_plan_agent, prompt)).final_output

            result = await Cache.instance().get_or_compute(
                query_plan_agent,
                prompt,
                QueryPlanModel,
                compute_query_plan,
            )

            query_plan = QueryPlan.map_query_plan_model_to_query_plan(result)
            span.span_data.data["query_plan"] = query_plan.to_span_data()
            return query_plan

    async def _replan_query_plan(
        self,
        query: str,
        old_query_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
        exception: ReplanningNeeded,
    ) -> QueryPlan:
        # TODO: ask all data agents for the information needs again for participation (that are not currently participating)
        QueryReplanModel = QueryPlan.create_query_replan_model(participating_agents)

        id_index = QueryPlan.collect_answered_needs(old_query_plan)

        participating_agents_str = "\n".join(
            [
                f"【{agent.name()}】\n {capability}"
                for agent, capability in participating_agents
            ]
        )

        exception_dict = exception.model_dump()
        exception_dict["query_node"] = exception.query_node.to_replan_context()

        instructions = load_prompt(
            "query_coordinator_query_replan_agent_instructions",
            None,
            {
                "participating_agents": participating_agents_str,
                "old_query_plan": json.dumps(
                    old_query_plan.to_replan_context(), indent=2
                ),
            },
        )
        prompt = load_prompt(
            "query_coordinator_query_replan_agent_prompt",
            None,
            {"query": query, "exception": json.dumps(exception_dict, indent=2)},
        )

        query_replan_agent = self.agent.clone(
            name=f"{self.agent.name}_Query_Replan_Agent",
            instructions=instructions,
            output_type=QueryReplanModel,
        )

        async def compute_query_replan():
            return (await Runner.run(query_replan_agent, prompt)).final_output

        result = await Cache.instance().get_or_compute(
            query_replan_agent,
            prompt,
            QueryReplanModel,
            compute_query_replan,
        )

        query_plan = QueryPlan.map_query_replan_model_to_query_plan(result, id_index)
        return query_plan

    async def _execute_query_plan(
        self,
        query: str,
        query_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
    ) -> JsonValue:
        plan_to_execute = query_plan

        MAX_REPLAN_ATTEMPTS = 2  # Max 5 replans

        for i in range(1, MAX_REPLAN_ATTEMPTS + 2):
            try:
                with custom_span(f"{i}: Execute Query Plan") as span:
                    print(f"{i}: Execute Query Plan")
                    span.span_data.data["query_plan"] = plan_to_execute.to_span_data()
                    span.span_data.data["participating_agents"] = [
                        agent.name() for agent, _ in participating_agents
                    ]

                    answer = await self._execute_query_plan_node(
                        plan_to_execute, participating_agents
                    )

                    append_json_preview_to_span("final_answer", answer, span)
                    return answer
            except ReplanningNeededException as exception:
                if i == MAX_REPLAN_ATTEMPTS + 1:
                    with custom_span(f"{i}: Max Replanning Attempts Reached") as span:
                        span.span_data.data["reason"] = exception.data.to_span_data()
                    break

                with custom_span(f"{i}: Replanning") as span:
                    print(f"{i}: Replanning")

                    exception_data = exception.data
                    span.span_data.data["reason"] = exception_data.to_span_data()
                    span.span_data.data["old_query_plan"] = (
                        plan_to_execute.to_span_data()
                    )

                    plan_to_execute = await self._replan_query_plan(
                        query, query_plan, participating_agents, exception_data
                    )

                    span.span_data.data["new_query_plan"] = (
                        plan_to_execute.to_span_data()
                    )

        return "Failed to answer the query after multiple attempts. The query might be too complex or there might be missing information that the agents cannot provide."

    async def _execute_query_plan_node(
        self, node: QueryPlan, participating_agents: list[tuple[DataAgent, str]]
    ) -> JsonValue:
        with custom_span(
            f"Execute Query Plan Node ({node.to_agent}): {node.question}"
        ) as span:
            print(f"    Execute Query Plan Node ({node.to_agent}): {node.question}")

            span.span_data.data["node"] = node.to_span_data()

            if node.answered_needs is None:
                node.answered_needs = []

            if node.needs is not None and len(node.needs) > 0:
                # Satisfy the information needs first
                for need in node.needs.copy():
                    answer = await self._execute_query_plan_node(
                        need, participating_agents
                    )
                    node.answered_needs.append(
                        Need(
                            need=need.need if need.need is not None else "",
                            answer=answer,
                        )
                    )
                    node.needs.remove(need)

            span.span_data.data["answered_needs"] = [
                need.to_span_data() for need in node.answered_needs
            ]

            if node.to_agent == "Coordinator":
                return await self._aggregate_results(node)

            responsible_data_agent = next(
                (
                    agent
                    for agent, _ in participating_agents
                    if agent.name() == node.to_agent
                ),
                None,
            )

            if responsible_data_agent is None:
                raise ReplanningNeededException(
                    data=ResponsibleAgentNotFound(query_node=node)
                )

            answered_needs = (
                node.answered_needs if node.answered_needs is not None else []
            )

            return await self._answer_query_plan_node_with_agent(
                node, responsible_data_agent, answered_needs
            )

    async def _answer_query_plan_node_with_agent(
        self, node: QueryPlan, agent: DataAgent, answered_needs: list[Need]
    ) -> JsonValue:
        with custom_span(
            f"Answer Query Plan Node with Agent {agent.name()}: {node.question}"
        ) as span:
            span.span_data.data["node"] = node.to_span_data()
            span.span_data.data["answered_needs"] = [
                answered_need.to_span_data() for answered_need in answered_needs
            ]

            answer = await agent.answer_query(node.question, answered_needs)
            span.span_data.data["answer"] = answer.to_span_data()

            if answer.needs is not None and len(answer.needs) > 0:
                raise ReplanningNeededException(
                    data=NeedNotMet(partial_answer=answer, query_node=node)
                )

            return answer.answer

    async def _aggregate_results(self, node: QueryPlan) -> JsonValue:
        with custom_span("Answer Query Plan Node") as span:
            needs: list[Any] = []
            answer: Any | Symbol = Symbol()

            additional_information = [
                {
                    "information": answered_need.need,
                    "data": answered_need.answer,
                }
                for answered_need in node.answered_needs or []
            ]

            def raise_needs(new_needs: list[Any]):
                needs.extend(new_needs)

            def final_answer(new_answer: Any):
                nonlocal answer
                answer = new_answer

            code_environment = CodeExecutionEnvironment(
                variables={
                    "query": node.question,
                    "additional_information": additional_information,
                },
                tools={
                    "final_answer": final_answer,
                },
            )

            instructions = load_prompt(
                "query_coordinator_aggregate_agent_instructions", None, {}
            )
            prompt = load_prompt(
                "query_coordinator_aggregate_agent_prompt",
                None,
                {"query": node.question},
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

            aggregate_agent = self.agent.clone(
                name=f"{self.agent.name}_Aggregate_Agent",
                instructions=instructions,
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                output_type=AgentOutputSchema(
                    PartialQueryAnswer, strict_json_schema=False
                ),
                reset_tool_choice=False,
                model_settings=dataclasses.replace(
                    self.agent.model_settings,
                    parallel_tool_calls=False,
                    tool_choice="required",
                ),
            )

            async def compute_aggregate_result() -> PartialQueryAnswer:
                return (await Runner.run(aggregate_agent, prompt)).final_output

            result = await compute_aggregate_result()
            # result = await Cache.instance().get_or_compute(
            #     aggregate_agent,
            #     prompt,
            #     JsonValue,
            #     compute_aggregate_result,
            # )

            span.span_data.data["node"] = node.to_span_data()
            span.span_data.data["answer"] = result.to_span_data()
            code_environment.append_data_to_span(span)

            if result.needs is not None and len(result.needs) > 0:
                raise ReplanningNeededException(
                    data=NeedNotMet(partial_answer=result, query_node=node)
                )

            return result.answer
