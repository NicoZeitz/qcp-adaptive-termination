import asyncio
import textwrap
from typing import Any, List, Literal, cast

from agents import Runner, custom_span, custom_span
from pydantic import PrivateAttr, create_model, Field

from modules.data_agent import DataAgent
from modules.query_tree import Need, QueryPlan

class ReplanningNeededException(Exception):
    pass

def relevance_agent_instructions(agent_registry: str) -> str:
    return textwrap.dedent(f"""Assess the relevance of each agent based on the following registry. A agent is relevant if it may be able to provide information that could be used to answer the question, even if only partially. The agent is relevant if it could help answering the question in any indirect, partial or abstract way as well. Only return False if there is no connection between the agent's capabilities and the question.

【Agent Registry】
{agent_registry}
""").strip()

def relevance_agent_prompt(query: str) -> str:
    return f"Is the query '{query}' relevant to any of the listed agents?"

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

def query_replan_agent_instructions(participating_agents: list[tuple[DataAgent, str]]) -> str:
    return query_plan_agent_instructions(participating_agents) + "\n\n" + textwrap.dedent("""
The initial plan was created on wrong assumptions and needs to be replanned. Please create a new plan based on the same rules.

""").strip() # TODO:

def query_replan_agent_prompt(query: str) -> str:
    return textwrap.dedent(f"""{query}""").strip() # TODO:

class QueryCoordinator:
    def __init__(self, agent):
        self.data_agents: list[DataAgent] = []
        self.agent = agent

    def register_data_agent(self, data_agent: DataAgent):
        self.data_agents.append(data_agent)

    def unregister_data_agent(self, data_agent: DataAgent):
        self.data_agents.remove(data_agent)

    async def collaborative_query_answering(self, query: str) -> str:
        agent_registry = await self._build_agent_registry()

        with custom_span("Collaborative Query Answering"):
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

    def _create_query_plan_model(self, participating_agents: list[tuple[DataAgent, str]]) -> QueryPlan:
        participating_agents_names = [agent.name() for agent, _ in participating_agents]
        QueryPlanModel = create_model(
            "QueryPlanModel",
            need=(str | None, Field(..., description="The specific information need that this question is addressing. This is used to connect the nodes to their parent's question in the query plan. Empty for the root node.")),
            question=(str, Field(..., description="The specific question that is answered by this node.")),
            from_agent=(Literal["User", "Coordinator"], Field(..., description="The agent that asked the question. This can be the User if it's the root node, or the Coordinator if it's a sub-question that needs to be answered in order to answer the parent question.")),
            to_agent=(Literal["Coordinator", *participating_agents_names], Field(..., description="The agent to which the question is directed.")),
            needs=(List["QueryPlanModel"] | None, Field(..., description="The sub-questions that need to be answered in order to answer the parent question. Should be empty if the question can be answered directly by an agent. Multiple needs are independent from each other. If there is a dependence between the needs, they should be nested accordingly to reflect the dependence relationship. ")),
            _answered_needs=(List[Need] | None, PrivateAttr(...,))
        )
        QueryPlanModel.model_rebuild()
        return cast(QueryPlan, QueryPlanModel)

    async def _draft_query_plan(self, query: str, participating_agents: list[tuple[DataAgent, str]]) -> QueryPlan:
        with custom_span("Draft Query Plan") as span:
            QueryPlanModel = self._create_query_plan_model(participating_agents)

            query_plan_agent = self.agent.clone(
                instructions=query_plan_agent_instructions(participating_agents),
                output_type=QueryPlanModel
            )

            result = await Runner.run(
                query_plan_agent,
                query_plan_agent_prompt(query)
            )

            span.span_data.data["query_plan"] = result.final_output
            return cast(QueryPlan, result.final_output)

    async def _execute_query_plan(self, query: str, query_plan: QueryPlan, participating_agents: list[tuple[DataAgent, str]]) -> str:
        plan_to_execute = query_plan

        i = 0
        while i < 5: # Max 5 replans
            try:
                with custom_span("Execute Query Plan") as span:
                    span.span_data.data["query_plan"] = plan_to_execute
                    span.span_data.data["participating_agents"] = [agent.name() for agent, _ in participating_agents]

                    answer = await self._execute_query_plan_node(plan_to_execute, participating_agents)

                    span.span_data.data["final_answer"] = answer
                    return answer
            except ReplanningNeededException as e:
                i += 1
                with custom_span("Replanning") as span:
                    span.span_data.data["reason"] = str(e)
                    plan_to_execute = await self._replan_query_plan(
                        query,
                        query_plan,
                        participating_agents
                    )


        return "Failed to answer the query after multiple attempts. The query might be too complex or there might be missing information that the agents cannot provide."

    async def _replan_query_plan(self, query: str, query_plan: QueryPlan, participating_agents: list[tuple[DataAgent, str]]) -> QueryPlan:
        with custom_span("Replan Query Plan") as span:
            QueryPlanModel = self._create_query_plan_model(participating_agents)

            query_plan_agent = self.agent.clone(
                instructions=query_replan_agent_instructions(participating_agents),
                output_type=QueryPlanModel
            )
            # TODO: ask all data agents for the information needs again for participation

            result = await Runner.run(
                query_plan_agent,
                query_replan_agent_prompt(query)
            )

            span.span_data.data["query_plan"] = result.final_output
            return cast(QueryPlan, result.final_output)

    async def _execute_query_plan_node(self, node: QueryPlan, participating_agents: list[tuple[DataAgent, str]]) -> str:

        if "_answered_needs" not in node.__dict__ or node._answered_needs is None:
            node._answered_needs = []

        if node.needs is not None and len(node.needs) > 0:
            # Satisfy the information needs first
            for need in node.needs.copy():
                answer = await self._execute_query_plan_node(need, participating_agents)
                node._answered_needs.append(
                    Need(need=need.need if need.need is not None else "", answer=answer)
                )
                node.needs.remove(need)


        if node.to_agent == "Coordinator":
            # TODO:
            raise NotImplementedError("Coordinator asking questions to the user is not implemented yet.")

        responsible_data_agent = next((agent for agent, _ in participating_agents if agent.name() == node.to_agent), None)

        if responsible_data_agent is None:
            raise ReplanningNeededException()

        _answered_needs = node._answered_needs if node._answered_needs is not None else []
        additional_information = "\n".join([f"{need.need}: {need.answer}" for need in _answered_needs])

        with custom_span("Answer Query Plan Node") as span:
            span.span_data.data["node"] = node.model_dump_json()
            span.span_data.data["additional_information"] = additional_information

            answer = await responsible_data_agent.answer_query(node.question, additional_information)
            if answer.needs is not None:
                raise ReplanningNeededException()

            return answer.answer