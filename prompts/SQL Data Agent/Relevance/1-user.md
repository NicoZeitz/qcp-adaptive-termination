The Coordinator thinks you might be relevant for this query.

【Query for network】
{{query}}

【Coordinator reasoning】
{{reasoning}}

**Your Task:**
Evaluate whether you can help answer this query or parts of it with your data. You are relevant if any part of the query could be answered using your data, even in a partial, indirect, or abstract way. Even loose or inferred connections count. Also factor the data the coordinator asks from you.


**Return True if:**
- Your schema contains the specific data the coordinator asks for
- You can retrieve the information (fully or partially) using SQL queries
- You have reasonable confidence you can help with the task

If you think there is no connection between the query and your data or your data is not useful in any way to answer the query: Look one more time at the reasoning of the coordinator and if there is as well no ask you can answer, return False.