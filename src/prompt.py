from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client

load_dotenv()

langfuse = get_client()

name_mapping = {
    "query_coordinator_aggregate_agent_instructions": "Query Coordinator/Aggregate/Instructions",
    "query_coordinator_aggregate_agent_prompt": "Query Coordinator/Aggregate/Prompt",
    "query_coordinator_local_replan_agent_instructions": "Query Coordinator/Local Replan/Instructions",
    "query_coordinator_local_replan_agent_prompt": "Query Coordinator/Local Replan/Prompt",
    "query_coordinator_query_plan_agent_instructions": "Query Coordinator/Query Plan/Instructions",
    "query_coordinator_query_plan_agent_prompt": "Query Coordinator/Query Plan/Prompt",
    "query_coordinator_query_replan_agent_instructions": "Query Coordinator/Query Replan/Instructions",
    "query_coordinator_query_replan_agent_prompt": "Query Coordinator/Query Replan/Prompt",
    "query_coordinator_relevance_agent_instructions": "Query Coordinator/Relevance/Instructions",
    "query_coordinator_relevance_agent_prompt": "Query Coordinator/Relevance/Prompt",
    "sql_data_agent_code_agent_instructions": "SQL Data Agent/Code/Instructions",
    "sql_data_agent_code_agent_prompt": "SQL Data Agent/Code/Prompt",
    "sql_data_agent_relevance_agent_instructions": "SQL Data Agent/Relevance/Instructions",
    "sql_data_agent_relevance_agent_prompt": "SQL Data Agent/Relevance/Prompt",
    "sql_data_agent_schema_description_agent_instructions": "SQL Data Agent/Schema Description/Instructions",
    "sql_data_agent_schema_description_agent_prompt": "SQL Data Agent/Schema Description/Prompt",
    "text_data_agent_code_agent_instructions": "Text Data Agent/Code/Instructions",
    "text_data_agent_code_agent_prompt": "Text Data Agent/Code/Prompt",
    "text_data_agent_relevance_agent_instructions": "Text Data Agent/Relevance/Instructions",
    "text_data_agent_relevance_agent_prompt": "Text Data Agent/Relevance/Prompt",
    "text_data_agent_summary_agent_instructions": "Text Data Agent/Summary/Instructions",
    "text_data_agent_summary_agent_prompt": "Text Data Agent/Summary/Prompt",
}

for key, value in name_mapping.items():
    path = f"../prompts/{key}.md"
    with open(path) as f:
        prompt = f.read().replace("{{{", "{{").replace("}}}", "}}")
        # print(prompt)
        langfuse.create_prompt(
            name=value,
            type="text",
            prompt=prompt
        )

# langfuse.create_prompt(
#     name="movie-critic",
#     type="text",
#     prompt="As a {{criticlevel}} movie critic, do you like {{movie}}?",
#     labels=["production"],  # add the label "production" to the prompt version
# )