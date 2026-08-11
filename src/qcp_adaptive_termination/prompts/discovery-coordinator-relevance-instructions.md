For each agent in the registry, determine:
1. **is_relevant**: Whether the agent might be able to help answer the query (even partially or indirectly)
2. **reasoning**: A clear and concise explanation of WHY you think the agent is or isn't relevant to help answer the query

The reasoning will be shared with the agent as they know their data better than you and can check if your assessment is correct.

Be specific about:
- What data the agent has that relates to the query
- How the agent could contribute to answering the query
- What assumptions you're making about the agent's usefulness and data

Additional remarks:
- Don't tell the agent what the broader plan beside his involvement is
- Just tell him: 'I need this, I think you have that, ...'
- Address the agent as 'you'

A agent is relevant if it can provide any information that could be used to answer the query even if only partially or in an indirect or abstract way. Only return False and mark the agent as irrelevant if the agents data cannot be used in any way to help th query answering. It is better to mark an agent falsely as relevant then later discovering, that the agent was needed to answer the query! A good measure for table based agents is for example, that if there is any column which could be necessary to answer the query, even only in context with other agents, to mark it as relevant.

【Agent Registry】
{{agent_registry}}
