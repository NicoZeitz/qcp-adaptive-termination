import asyncio
from typing import Any, cast

from agents import Runner, custom_span
from pydantic import create_model, Field

from modules.cache import Cache
from modules.model import (
    DataAgent,
    Need,
    NeedNotMet,
    QueryPlan,
    ReplanningNeeded,
    ReplanningNeededException,
    ResponsibleAgentNotFound,
)
from prompt import load_prompt


class QueryCoordinator:
    def __init__(self, agent):
        self.data_agents: list[DataAgent] = []
        self.agent = agent

    def register_data_agent(self, data_agent: DataAgent):
        print("Register data agent:", data_agent.name())
        self.data_agents.append(data_agent)

    def unregister_data_agent(self, data_agent: DataAgent):
        print("Unregister data agent:", data_agent.name())
        self.data_agents.remove(data_agent)

    async def collaborative_query_answering(self, query: str) -> str:
        agent_registry = await self._build_agent_registry()

        with custom_span("Collaborative Query Answering") as span:
            span.span_data.data["agent_registry"] = agent_registry
            span.span_data.data["query"] = query

            # 1. Query Coordinator identifies Relevant Agents Based on Capabilities.
            relevant_agents = await self._filter_relevant_agents(query, agent_registry)

            # 2. Data Agents are Asked if they participate
            participating_agents = await self._ask_data_agents_for_participation(
                relevant_agents, query
            )

            # 3. Coordinator Drafts Query Plan & Initializes Stack Based on Previously Identified Agents.
            query_plan = await self._draft_query_plan(query, participating_agents)

            result = await self._execute_query_plan(
                query, query_plan, participating_agents
            )
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

            span.span_data.data["query_plan"] = query_plan.model_dump()
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

        participating_agents_str = "\n".join(
            [
                f"【{agent.name()}】\n {capability}"
                for agent, capability in participating_agents
            ]
        )

        instructions = load_prompt(
            "query_coordinator_query_replan_agent_instructions",
            None,
            {
                "participating_agents": participating_agents_str,
                "old_query_plan": old_query_plan.model_dump_json(indent=2),
            },
        )
        prompt = load_prompt(
            "query_coordinator_query_replan_agent_prompt",
            None,
            {"query": query, "exception": exception.model_dump_json(indent=2)},
        )

        query_plan_agent = self.agent.clone(
            name=f"{self.agent.name}_Query_Replan_Agent",
            instructions=instructions,
            output_type=QueryReplanModel,
        )

        async def compute_query_replan():
            return (await Runner.run(query_plan_agent, prompt)).final_output

        result = await Cache.instance().get_or_compute(
            query_plan_agent,
            prompt,
            QueryReplanModel,
            compute_query_replan,
        )

        query_plan = QueryPlan.map_query_replan_model_to_query_plan(result)
        return query_plan

    async def _execute_query_plan(
        self,
        query: str,
        query_plan: QueryPlan,
        participating_agents: list[tuple[DataAgent, str]],
    ) -> str:
        plan_to_execute = query_plan

        MAX_REPLAN_ATTEMPTS = 2  # Max 5 replans

        for i in range(1, MAX_REPLAN_ATTEMPTS + 2):
            try:
                with custom_span(f"{i}: Execute Query Plan") as span:
                    print(f"{i}: Execute Query Plan")
                    span.span_data.data["query_plan"] = plan_to_execute.model_dump()
                    span.span_data.data["participating_agents"] = [
                        agent.name() for agent, _ in participating_agents
                    ]

                    answer = await self._execute_query_plan_node(
                        plan_to_execute, participating_agents
                    )

                    span.span_data.data["final_answer"] = answer
                    return answer
            except ReplanningNeededException as exception:
                if i == MAX_REPLAN_ATTEMPTS:
                    with custom_span(f"{i}: Max Replanning Attempts Reached") as span:
                        span.span_data.data["reason"] = exception.data.model_dump()
                    break

                with custom_span(f"{i}: Replanning") as span:
                    print(f"{i}: Replanning")

                    exception_data = exception.data
                    span.span_data.data["reason"] = exception_data.model_dump()
                    span.span_data.data["old_query_plan"] = plan_to_execute.model_dump()

                    plan_to_execute = await self._replan_query_plan(
                        query, query_plan, participating_agents, exception_data
                    )

                    span.span_data.data["new_query_plan"] = plan_to_execute.model_dump()

        return "Failed to answer the query after multiple attempts. The query might be too complex or there might be missing information that the agents cannot provide."

    async def _execute_query_plan_node(
        self, node: QueryPlan, participating_agents: list[tuple[DataAgent, str]]
    ) -> str:
        with custom_span(
            f"Execute Query Plan Node ({node.to_agent}): {node.question}"
        ) as span:
            print(f"    Execute Query Plan Node ({node.to_agent}): {node.question}")

            span.span_data.data["node"] = node.model_dump()

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
                need.model_dump() for need in node.answered_needs
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
    ) -> str:
        with custom_span(
            f"Answer Query Plan Node with Agent {agent.name()}: {node.question}"
        ) as span:
            span.span_data.data["node"] = node.model_dump()
            span.span_data.data["answered_needs"] = [
                answered_need.model_dump() for answered_need in answered_needs
            ]

            answer = await agent.answer_query(node.question, answered_needs)
            span.span_data.data["answer"] = answer.model_dump()

            if answer.needs is not None:
                raise ReplanningNeededException(
                    data=NeedNotMet(partial_answer=answer, query_node=node)
                )

            return answer.answer

    async def _aggregate_results(self, node: QueryPlan) -> str:
        with custom_span("Answer Query Plan Node") as span:
            span.span_data.data["node"] = node.model_dump()

            answered_needs = (
                node.answered_needs if node.answered_needs is not None else []
            )
            additional_information = "\n".join(
                [f"{need.need}: {need.answer}" for need in answered_needs]
            )

            instructions = load_prompt(
                "query_coordinator_aggregate_agent_instructions", None, {}
            )
            prompt = load_prompt(
                "query_coordinator_aggregate_agent_prompt",
                None,
                {
                    "answered_needs": answered_needs,
                    "additional_information": additional_information,
                },
            )

            aggregate_agent = self.agent.clone(
                name=f"{self.agent.name}_Aggregate_Agent",
                instructions=instructions,
            )

            async def compute_aggregate_result():
                return cast(
                    str,
                    (await Runner.run(aggregate_agent, prompt)).final_output,
                )

            result = cast(
                str,
                await Cache.instance().get_or_compute(
                    aggregate_agent,
                    prompt,
                    str,
                    compute_aggregate_result,
                ),
            )

            span.span_data.data["result"] = result
            return result
