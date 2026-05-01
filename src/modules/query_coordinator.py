import asyncio
import dataclasses
import json
from typing import Any, cast

from agents import (
    Agent,
    AgentOutputSchema,
    FunctionToolResult,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
)
from pydantic import JsonValue, create_model, Field

from modules.code_execution_environment import CodeExecutionEnvironment
from modules.model import (
    CHAR_LIMIT_SPAN,
    AgentContext,
    DataAgent,
    Need,
    NeedNotMet,
    PartialQueryAnswer,
    QueryPlan,
    ReplanMode,
    ReplanningNeeded,
    ResponsibleAgentNotFound,
    Symbol,
    append_to_span_input,
    append_to_span_output,
    own_custom_span,
    preview_text,
    serialize_value,
)
from modules.prompt import load_prompt


class QueryCoordinator:
    def __init__(
        self,
        agent: Agent[AgentContext],
        replan_mode: ReplanMode = ReplanMode.GLOBAL,
    ):
        self.data_agents: list[DataAgent] = []
        self.agent: Agent[AgentContext] = agent
        self.replan_mode = replan_mode

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
        with own_custom_span("Build Agent Registry") as span:
            capabilities = await asyncio.gather(
                *(agent.get_data_and_query_capabilities() for agent in self.data_agents)
            )

            agent_registry = ""
            for agent, capability in zip(self.data_agents, capabilities):
                append_to_span_output(span, {f"{agent.name()}_capability": capability})
                agent_registry += f"{agent.name()}: {capability}\n"

            append_to_span_output(span, {"agent_registry": agent_registry})

            return agent_registry.strip()

    async def _filter_relevant_agents(
        self, query: str, agent_registry: str
    ) -> list[DataAgent]:
        with own_custom_span("Filtering Relevant Agents") as span:
            agent_fields: dict[str, Any] = {
                agent.name(): (bool, Field(...)) for agent in self.data_agents
            }
            RelevanceModel = create_model("RelevanceModel", **agent_fields)

            instructions = load_prompt(
                "Query Coordinator/Relevance/Instructions",
                {"agent_registry": agent_registry},
            )
            prompt = load_prompt("Query Coordinator/Relevance/Prompt", {"query": query})

            relevance_agent = self.agent.clone(
                name=f"{self.agent.name}_Relevance_Agent",
                instructions=instructions,
                output_type=RelevanceModel,
            )

            relevances = (await Runner.run(relevance_agent, prompt)).final_output

            relevant_agents = []
            for agent in self.data_agents:
                relevant = getattr(relevances, agent.name(), False)

                append_to_span_output(span, {f"{agent.name()}_relevant": relevant})
                if relevant:
                    relevant_agents.append(agent)

            return relevant_agents

    async def _ask_data_agents_for_participation(
        self, relevant_agents: list[DataAgent], query: str
    ) -> list[tuple[DataAgent, str]]:
        with own_custom_span("Asking Data Agents for Participation") as span:
            schemas = await asyncio.gather(
                *(agent.get_schema_for_query(query) for agent in relevant_agents)
            )

            participating_agents = []
            for agent, schema in zip(relevant_agents, schemas):
                relevant = schema is not None

                append_to_span_output(span, {f"{agent.name()}_relevant": relevant})
                if relevant:
                    participating_agents.append((agent, schema))

            return participating_agents

    async def _draft_query_plan(
        self, query: str, participating_agents: list[tuple[DataAgent, str]]
    ) -> QueryPlan:
        with own_custom_span("Draft Query Plan") as span:
            QueryPlanModel = QueryPlan.create_query_plan_model(participating_agents)
            participating_agents_str = "\n".join(
                [
                    f"【{agent.name()}】\n {capability}"
                    for agent, capability in participating_agents
                ]
            )

            instructions = load_prompt(
                "Query Coordinator/Query Plan/Instructions",
                {"participating_agents": participating_agents_str},
            )
            prompt = load_prompt(
                "Query Coordinator/Query Plan/Prompt", {"query": query}
            )

            query_plan_agent = self.agent.clone(
                name=f"{self.agent.name}_Query_Plan_Agent",
                instructions=instructions,
                output_type=QueryPlanModel,
            )

            result = (await Runner.run(query_plan_agent, prompt)).final_output
            query_plan = QueryPlan.map_query_plan_model_to_query_plan(result)
            append_to_span_output(span, {"query_plan": query_plan.to_span_data()})
            return query_plan

    async def _replan_query_plan(
        self,
        query: str,
        current_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
        failure: ReplanningNeeded,
    ) -> QueryPlan:
        # TODO: ask all data agents for the information needs again for participation (that are not currently participating)
        QueryReplanModel = QueryPlan.create_query_replan_model(participating_agents)

        id_index = QueryPlan.collect_answered_needs(current_plan)

        participating_agents_str = "\n".join(
            [
                f"【{agent.name()}】\n {capability}"
                for agent, capability in participating_agents
            ]
        )

        failure_dict = failure.model_dump()
        failure_dict["query_node"] = failure.query_node.to_replan_context()

        instructions = load_prompt(
            "Query Coordinator/Query Replan/Instructions",
            {
                "participating_agents": participating_agents_str,
                "old_query_plan": json.dumps(
                    current_plan.to_replan_context(), indent=2
                ),
            },
        )
        prompt = load_prompt(
            "Query Coordinator/Query Replan/Prompt",
            {"query": query, "exception": json.dumps(failure_dict, indent=2)},
        )

        query_replan_agent = self.agent.clone(
            name=f"{self.agent.name}_Query_Replan_Agent",
            instructions=instructions,
            output_type=QueryReplanModel,
        )

        result = (await Runner.run(query_replan_agent, prompt)).final_output
        return QueryPlan.map_query_replan_model_to_query_plan(result, id_index)

    async def _replan_query_plan_local(
        self,
        query: str,
        current_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
        failure: ReplanningNeeded,
    ) -> QueryPlan:
        failing_node = failure.query_node
        QueryReplanModel = QueryPlan.create_query_replan_model(participating_agents)

        id_index = QueryPlan.collect_answered_needs(failing_node)

        participating_agents_str = "\n".join(
            [
                f"【{agent.name()}】\n {capability}"
                for agent, capability in participating_agents
            ]
        )

        failure_dict = failure.model_dump()
        failure_dict["query_node"] = failing_node.to_replan_context()

        instructions = load_prompt(
            "Query Coordinator/Local Replan/Instructions",
            {
                "participating_agents": participating_agents_str,
                "failing_node": json.dumps(failing_node.to_replan_context(), indent=2),
            },
        )
        prompt = load_prompt(
            "Query Coordinator/Local Replan/Prompt",
            {"query": query, "exception": json.dumps(failure_dict, indent=2)},
        )

        local_replan_agent = self.agent.clone(
            name=f"{self.agent.name}_Local_Replan_Agent",
            instructions=instructions,
            output_type=QueryReplanModel,
        )

        result = (await Runner.run(local_replan_agent, prompt)).final_output
        new_subtree = QueryPlan.map_query_replan_model_to_query_plan(result, id_index)
        return QueryPlan.replace_node(current_plan, failing_node.id, new_subtree)

    async def _execute_query_plan(
        self,
        query: str,
        query_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
    ) -> JsonValue:
        plan_to_execute = query_plan

        MAX_REPLAN_ATTEMPTS = 2

        for i in range(1, MAX_REPLAN_ATTEMPTS + 2):
            with own_custom_span(f"{i}: Execute Query Plan") as span:
                append_to_span_input(span, {
                    "query_plan": plan_to_execute.to_span_data(),
                    "participating_agents": [agent.name() for agent, _ in participating_agents]
                })

                result = await self._execute_query_plan_node(
                    plan_to_execute, participating_agents
                )

                if not isinstance(result, ReplanningNeeded):
                    append_to_span_output(span, {"final_answer": preview_text(str(result), CHAR_LIMIT_SPAN)})
                    return result

                execution_failure = result
                append_to_span_output(span, {"failure": execution_failure.to_span_data()})

            if i == MAX_REPLAN_ATTEMPTS + 1:
                with own_custom_span(f"{i}: Max Replanning Attempts Reached") as span:
                    append_to_span_output(span, {"reason": execution_failure.to_span_data()})
                break

            with own_custom_span(f"{i}: Replanning ({self.replan_mode.value})") as span:
                append_to_span_input(span, {
                    "reason": execution_failure.to_span_data(),
                    "old_query_plan": plan_to_execute.to_span_data()
                })


                if self.replan_mode == ReplanMode.GLOBAL:
                    plan_to_execute = await self._replan_query_plan(
                        query, plan_to_execute, participating_agents, execution_failure
                    )
                else:
                    plan_to_execute = await self._replan_query_plan_local(
                        query, plan_to_execute, participating_agents, execution_failure
                    )

                append_to_span_output(span, {"new_query_plan": plan_to_execute.to_span_data()})

        return "Failed to answer the query after multiple attempts. The query might be too complex or there might be missing information that the agents cannot provide."

    async def _execute_query_plan_node(
        self, node: QueryPlan, participating_agents: list[tuple[DataAgent, str]]
    ) -> JsonValue | ReplanningNeeded:
        with own_custom_span(
            f"Execute Query Plan Node ({node.to_agent}): {node.question}"
        ) as span:
            append_to_span_input(span, {"node": node.to_span_data()})

            if node.answered_needs is None:
                node.answered_needs = []

            if node.needs is not None and len(node.needs) > 0:
                # Sibling needs are independent — execute them in parallel
                needs_snapshot = node.needs.copy()
                answers = await asyncio.gather(
                    *(
                        self._execute_query_plan_node(need, participating_agents)
                        for need in needs_snapshot
                    )
                )

                # Bubble up the first failure — don't process any results
                for answer in answers:
                    if isinstance(answer, ReplanningNeeded):
                        return answer

                for need, answer in zip(needs_snapshot, answers):
                    node.answered_needs.append(
                        Need(
                            need=need.need if need.need is not None else "",
                            answer=cast(JsonValue, answer),
                        )
                    )
                node.needs.clear()


            append_to_span_input(span, {"answered_needs": [need.to_span_data() for need in node.answered_needs]})

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
                return ResponsibleAgentNotFound(query_node=node)

            answered_needs = (
                node.answered_needs if node.answered_needs is not None else []
            )

            return await self._answer_query_plan_node_with_agent(
                node, responsible_data_agent, answered_needs
            )

    async def _answer_query_plan_node_with_agent(
        self, node: QueryPlan, agent: DataAgent, answered_needs: list[Need]
    ) -> JsonValue | ReplanningNeeded:
        with own_custom_span(
            f"Answer Query Plan Node with Agent {agent.name()}: {node.question}"
        ) as span:
            append_to_span_input(span, {"node": node.to_span_data()})

            answer = await agent.answer_query(node.question, answered_needs)

            append_to_span_output(span, {"partial_answer": answer.to_span_data()})

            if answer.needs is not None and len(answer.needs) > 0:
                return NeedNotMet(partial_answer=answer, query_node=node)

            return answer.answer

    async def _aggregate_results(self, node: QueryPlan) -> JsonValue | ReplanningNeeded:
        with own_custom_span(
            f"Answer Query Plan Node with {self.agent.name}: {node.question}"
        ) as span:
            append_to_span_input(span, { "node": node.to_span_data() })

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
                    "raise_needs": raise_needs,
                },
            )

            instructions = load_prompt("Query Coordinator/Aggregate/Instructions", {})
            prompt = load_prompt(
                "Query Coordinator/Aggregate/Prompt",
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

            result = cast(
                PartialQueryAnswer,
                (await Runner.run(aggregate_agent, prompt)).final_output,
            )

            append_to_span_output(span, {
                "code": code_environment.to_span_data(),
                "answer": result.to_span_data()
            })

            if result.needs is not None and len(result.needs) > 0:
                return NeedNotMet(partial_answer=result, query_node=node)

            return result.answer
