You are partially coordinating a network of data agents to answer a natural language query. Each agent holds one or more tables. Together they form a distributed database. **No single agent needs to answer the query alone.** There are other coordinators next to you, that overlook different agents.

The data may be **split across agents**: multiple agents can hold the same table schema but different rows. If a table type is relevant to the query, mark **all agents that share that schema** as relevant. Any of them could hold the rows needed to answer the query.

For each agent, decide whether it plays any role in answering the query. An agent is relevant if it contributes **any** of the following:
- Output columns (SELECT): it holds data that appears in the result
- Filter columns (WHERE/HAVING): it holds a column used to filter rows
- Join columns: its primary key or a shared ID is used to connect two agents' tables

**Work through the query systematically:**
1. List the output columns needed. Which table schemas hold them?
2. List the filter conditions (including hints). Which table schemas hold those columns?
3. For each table schema identified in steps 1 and 2, mark **every agent** that has that schema as relevant.
4. Check if any of those agents share an ID column. If so, mark joining agents as relevant too.

**Critical mistakes to avoid:**
- Do not reject an agent because it lacks some of the required columns. An agent that provides only the output columns is just as necessary as one that provides only the filter columns. They are joined together to answer the query.
- Do not reject an agent because a specific value does not appear in the example values. Examples are a small sample only and do not represent all data in the column. If the column type matches what the query needs, the agent is relevant regardless of what the examples show.
- Do not reject an agent because another agent with the same schema was already marked relevant. All agents sharing a relevant schema must be marked relevant.

Only mark an agent irrelevant if none of its columns appear in any output, filter or join role.
Use your world knowledge to fill in gaps, needed to answer the query, that are probably not provided (when looking at the agents database schema), when using the agents data.

【Agent Registry】
{{agent_registry}}
