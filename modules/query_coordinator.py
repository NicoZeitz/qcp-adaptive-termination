import asyncio
import textwrap
from typing import Any, cast

from agents import Runner, custom_span, custom_span
from pydantic import create_model, Field

from modules.model import DataAgent, Need, NeedNotMet, QueryPlan, ReplanningNeeded, ReplanningNeededException, ResponsibleAgentNotFound

def relevance_agent_instructions(agent_registry: str) -> str:
    return textwrap.dedent(f"""Assess the relevance of each agent based on the following registry. A agent is relevant if it may be able to provide information that could be used to answer the question, even if only partially. The agent is relevant if it could help answering the question in any indirect, partial or abstract way as well. Only return False if there is no connection between the agent's capabilities and the question.

【Agent Registry】
{agent_registry}
""").strip()

def relevance_agent_prompt(query: str) -> str:
    return f"Is the query '{query}' relevant to any of the listed agents?"

# TODO: fix don't force sql instructions onto agents for question
def query_plan_agent_instructions(participating_agents: list[tuple[DataAgent, str]]) -> str:
    participating_agents_str = "\n".join([f"【{agent.name()}】\n {capability}" for agent, capability in participating_agents])

    return textwrap.dedent("""
You are a query engine responsible for drafting a query plan to answer the user's question.
A query plan is a tree of information needs, where the root node is the user's original question, and each child node is a sub-question that needs to be answered in order to answer the parent question. The leaf nodes are questions that can be answered directly by the participating agents.

Rules to follow:
* Break the question down into as many sub-questions as needed to allow for direct answers from the participating agents.
* Always try to address questions to the agents instead of the 'Coordinator', as invoking the coordinator is more costly than invoking agents.
* Both agents and the coordinator can ask sub-questions to other agents to provide further context that is missing and is needed to answer the parent question.
* If a need can possible be answered by multiple agents, include all of them as children nodes.
* Children nodes are independent from each other and don't have access to each other's information, so if there is a dependence between the needs, they should be nested accordingly to reflect the dependence relationship.
* Questions should be instructed in natural language instead of a technical language like SQL

Here is a simple example of a query plan for the question "What's the population of New York, Beijing and London?":

A1 - North American Population Information by CityId
A2 - European Population Information by City Name
A3 - Asian Population Information by CityId + CityId and Name Mapping Information
A4 - CityId and Name Mapping Information

```json
{
    "need": null,
    "question": "What's the population of New York, Beijing and London?",
    "from_agent": "User",
    "to_agent": "Coordinator",
    "needs": [
        {
            "need": "Population in New York",
            "question": "What's the population of New York?",
            "from_agent": "Coordinator",
            "to_agent": "A1",
            "needs": [
                {
                    "need": "CityId for New York",
                    "question": "What's the CityId for New York?",
                    "from_agent": "Coordinator",
                    "to_agent": "A3",
                    "needs": []
                },
                {
                   "need": "CityId for New York",
                    "question": "What's the CityId for New York?",
                    "from_agent": "Coordinator",
                    "to_agent": "A4",
                    "needs": []
                }
            ]
        },
        {
            "need": "Population in London",
            "question": "What's the population of London?",
            "from_agent": "Coordinator",
            "to_agent": "A2",
            "needs": []
        },
        {
            "need": "Population in Beijing",
            "question": "What's the population of Beijing?",
            "from_agent": "Coordinator",
            "to_agent": "A3",
            "needs": [
                {
                    "need": "CityId for Beijing",
                    "question": "What's the CityId for Beijing?",
                    "from_agent": "Coordinator",
                    "to_agent": "A4",
                    "needs": []
                }
            ]
        }
    ]
}
```

The following agents have volunteered to participate in answering the question, and here are their capabilities:

""" + participating_agents_str).strip()

def query_plan_agent_prompt(query: str) -> str:
    return textwrap.dedent(f"""{query}""").strip()

def query_replan_agent_instructions(participating_agents: list[tuple[DataAgent, str]], old_query_plan: QueryPlan) -> str:
    return query_plan_agent_instructions(participating_agents) + "\n\n" + textwrap.dedent(f"""
The initial plan was created on wrong assumptions and needs to be replanned. Please create a new plan based on the same rules.

Here is the initial plan that needs to be replanned:

{old_query_plan.model_dump()}
""").strip()

def query_replan_agent_prompt(query: str, exception: ReplanningNeeded) -> str:
    return textwrap.dedent(f"""{query}

Here is a more detailed description of the reason why the initial plan failed and needs to be replanned:
{exception.model_dump()}
""").strip()

def aggregate_agent_instructions() -> str:
    return textwrap.dedent("""You are an agent responsible for aggregating the answers to the sub-questions in order to answer the parent question. Your task is to provide a final answer to the user's original question based on the answers to the sub-questions.""").strip()

def aggregate_agent_prompt(node: QueryPlan) -> str:
    answered_needs = node.answered_needs if node.answered_needs is not None else []
    additional_information = "\n".join([f"{need.need}: {need.answer}" for need in answered_needs])

    return textwrap.dedent(f"""Please provide a final answer to the user's original question based on the following additional information obtained from answering the sub-questions in the query plan:
{additional_information}
""").strip()

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
            relevant_agents = self.data_agents # TODO: remove
            # relevant_agents = await self._filter_relevant_agents(query, agent_registry)

            # 2. Data Agents are Asked if they participate
            participating_agents = await self._ask_data_agents_for_participation(relevant_agents, query)

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

            return agent_registry.strip()

    async def _filter_relevant_agents(self, query: str, agent_registry: str) -> list[DataAgent]:
        with custom_span("Filtering Relevant Agents") as span:
            agent_fields: dict[str, Any] = {agent.name(): (bool, Field(...)) for agent in self.data_agents}
            RelevanceModel = create_model("RelevanceModel", **agent_fields)

            relevance_agent = self.agent.clone(
                instructions=relevance_agent_instructions(agent_registry),
                output_type=RelevanceModel
            )

            relevances = (await Runner.run(
                relevance_agent,
                relevance_agent_prompt(query)
            )).final_output

            relevant_agents = []
            for agent in self.data_agents:
                relevant = relevances.__dict__.get(agent.name(), False)

                span.span_data.data[f"{agent.name()}_relevant"] = relevant
                if relevant:
                    relevant_agents.append(agent)

            return relevant_agents

    async def _ask_data_agents_for_participation(self, relevant_agents: list[DataAgent], query: str) -> list[tuple[DataAgent, str]]:
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

    async def _draft_query_plan(self, query: str, participating_agents: list[tuple[DataAgent, str]]) -> QueryPlan:
        with custom_span("Draft Query Plan") as span:
            query_plan_agent = self.agent.clone(
                instructions=query_plan_agent_instructions(participating_agents),
                output_type=QueryPlan.create_query_plan_model(participating_agents)
            )

            result = await Runner.run(
                query_plan_agent,
                query_plan_agent_prompt(query)
            )

            query_plan = QueryPlan.map_query_plan_model_to_query_plan(result.final_output)

            span.span_data.data["query_plan"] = query_plan.model_dump()
            return query_plan

    async def _replan_query_plan(self, query: str, old_query_plan: QueryPlan, participating_agents: list[tuple[DataAgent, str]], exception: ReplanningNeeded) -> QueryPlan:
        query_plan_agent = self.agent.clone(
            instructions=query_replan_agent_instructions(participating_agents, old_query_plan),
            output_type=QueryPlan.create_query_replan_model(participating_agents)
        )
        # TODO: ask all data agents for the information needs again for participation (that are not currently participating)

        result = await Runner.run(
            query_plan_agent,
            query_replan_agent_prompt(query, exception)
        )

        query_plan = QueryPlan.map_query_replan_model_to_query_plan(result.final_output)
        return query_plan

    async def _execute_query_plan(self, query: str, query_plan: QueryPlan, participating_agents: list[tuple[DataAgent, str]]) -> str:
        plan_to_execute = query_plan

        MAX_REPLAN_ATTEMPTS = 2 # Max 5 replans

        for i in range(1, MAX_REPLAN_ATTEMPTS+2):
            try:
                with custom_span(f"{i}: Execute Query Plan") as span:
                    print(f"{i}: Execute Query Plan")
                    span.span_data.data["query_plan"] = plan_to_execute.model_dump()
                    span.span_data.data["participating_agents"] = [agent.name() for agent, _ in participating_agents]

                    answer = await self._execute_query_plan_node(plan_to_execute, participating_agents)

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
                        query,
                        query_plan,
                        participating_agents,
                        exception_data
                    )

                    span.span_data.data["new_query_plan"] = plan_to_execute.model_dump()


        return "Failed to answer the query after multiple attempts. The query might be too complex or there might be missing information that the agents cannot provide."

    async def _execute_query_plan_node(self, node: QueryPlan, participating_agents: list[tuple[DataAgent, str]]) -> str:
        with custom_span(f"Execute Query Plan Node ({node.to_agent}): {node.question}") as span:
            print(f"    Execute Query Plan Node ({node.to_agent}): {node.question}")

            span.span_data.data["node"] = node.model_dump()

            if node.answered_needs is None:
                node.answered_needs = []

            if node.needs is not None and len(node.needs) > 0:
                # Satisfy the information needs first
                for need in node.needs.copy():
                    answer = await self._execute_query_plan_node(need, participating_agents)
                    node.answered_needs.append(
                        Need(need=need.need if need.need is not None else "", answer=answer)
                    )
                    node.needs.remove(need)

            span.span_data.data["answered_needs"] = [need.model_dump() for need in node.answered_needs]

            if node.to_agent == "Coordinator":
                return await self._aggregate_results(node)

            responsible_data_agent = next((agent for agent, _ in participating_agents if agent.name() == node.to_agent), None)

            if responsible_data_agent is None:
                raise ReplanningNeededException(data=ResponsibleAgentNotFound(query_node=node))

            answered_needs = node.answered_needs if node.answered_needs is not None else []
            additional_information = "\n".join([f"{need.need}: {need.answer}" for need in answered_needs])

            return await self._answer_query_plan_node_with_agent(node, responsible_data_agent, additional_information)

    async def _answer_query_plan_node_with_agent(self, node: QueryPlan, agent: DataAgent, additional_information: str) -> str:
         with custom_span(f"Answer Query Plan Node with Agent {agent.name()}: {node.question}") as span:
            span.span_data.data["node"] = node.model_dump()
            span.span_data.data["additional_information"] = additional_information

            answer = await agent.answer_query(node.question, additional_information)
            span.span_data.data["answer"] = answer.model_dump()

            if answer.needs is not None:
                    raise ReplanningNeededException(data=NeedNotMet(partial_answer=answer, query_node=node))

            return answer.answer

    async def _aggregate_results(self, node: QueryPlan) -> str:
        with custom_span("Answer Query Plan Node") as span:
            span.span_data.data["node"] = node.model_dump()

            aggregate_agent = self.agent.clone(
                instructions=aggregate_agent_instructions()
            )

            result = cast(str, (await Runner.run(
                aggregate_agent,
                aggregate_agent_prompt(node)
            )).final_output)

            span.span_data.data["result"] = result
            return result