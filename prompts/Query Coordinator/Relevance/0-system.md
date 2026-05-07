For each agent in the registry, determine:
1. **is_relevant**: Whether the agent might be able to help answer the query (even partially or indirectly)
2. **reasoning**: A clear and concise explanation of WHY you think the agent is or isn't relevant

The reasoning will be shared with the agent as they know their data better than you and can answer if your assessment is correct or which data they .
Be specific about:
- What data the agent has that relate to the query
- How the agent could contribute to answering the query
- What assumptions you're making about the agent's usefulness and data

Additional remarks:
- Don't tell the agent what the broader plan beside his involvement is
- Just tell him: 'I need this, I think you have that, ...'
- Address the agent as 'you'

Assess the relevance of each agent based on the following registry. A agent is relevant if it may be able to provide any information that could be used to answer the query, even if only partially. The agent is relevant if it could help answering the query in any indirect, partial or abstract way as well. Only return False if there is no possible connection between the agent's capabilities and the query.

【Agent Registry】
{{agent_registry}}