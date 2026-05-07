For each agent in the registry, determine:
1. **is_relevant**: Whether the agent might be able to help answer the query (even partially or indirectly)
2. **reasoning**: A clear explanation of WHY you think the agent is or isn't relevant

The reasoning will be shared with the agent so they can evaluate whether they agree with your assessment. Be specific about:
- What data/capabilities the agent has that relate to the query
- How the agent could contribute to answering the query
- What assumptions you're making about the agent's usefulness

Assess the relevance of each agent based on the following registry. A agent is relevant if it may be able to provide information that could be used to answer the question, even if only partially. The agent is relevant if it could help answering the question in any indirect, partial or abstract way as well. Only return False if there is no connection between the agent's capabilities and the question.

【Agent Registry】
{{agent_registry}}