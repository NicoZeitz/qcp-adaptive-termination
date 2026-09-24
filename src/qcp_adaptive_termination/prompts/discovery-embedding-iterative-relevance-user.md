【Original Query】
{{query}}

【Context (schemas of already confirmed relevant agents, empty if none found yet)】
{{context_tables}}

【Sub-queries searched but no relevant database found yet (with fail count)】
{{unresolved_sub_queries}}

【Candidate Agent: {{agent_name}}】
{{schema}}

Evaluate this candidate agent and decide: is it relevant, is the query now fully answerable, and what information (if any) is still missing?
For sub-queries that have failed multiple times, consider whether: (a) the search phrase should be rewritten to find the right database, or (b) that information is not available in any database and can be answered from world knowledge instead — in which case do not include it in missing_sub_queries and mark is_complete accordingly.
