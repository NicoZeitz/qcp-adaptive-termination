The Coordinator thinks you might be relevant for this query.

【Query for network】
{{query}}

【Coordinator reasoning】
{{reasoning}}

**Your Task:**
Evaluate whether you can help answer this query including the coordinator reasoning with your data. You are relevant if any part of the query or reasoning could be answered using your data, even in a partial, indirect, or abstract way. Even loose or inferred connections count.


**Return True if:**
- Your schema contains the specific data the coordinator asks for
- You can retrieve the information (fully or partially) using SQL queries
- You may be needed in some form to join information
- You may be able to provide additional information that is not explicitly asked for but could be useful to answer the query
- You have some confidence you can help with the task

If you think there is absolutely no connection between the query and your data or your data is not useful in any possible way to answer the query:
- Look one more time at the reasoning of the coordinator
- Rethink your answer again with the additional reasoning
- If the answer is still no (and only in this case), only then return False

It is preferable to flag yourself as potentially relevant, **even when uncertain**, than to later discover that your relevance should have been acknowledged before.