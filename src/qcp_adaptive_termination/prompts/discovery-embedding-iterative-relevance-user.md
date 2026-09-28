【Original query】
{{query}}{{#context_tables}}

【Context】
{{context_tables}}{{/context_tables}}{{#unresolved_sub_queries}}

【Failed sub-queries with count】
{{unresolved_sub_queries}}{{/unresolved_sub_queries}}{{#matched_query}}

【Sub-query that found this candidate agent】
{{matched_query}}{{/matched_query}}

【Candidate agent】
{{schema}}

Decide:
1. Is the candidate agents database actually relevant to the original query{{#matched_query}} and the search query that retrieved it{{/matched_query}}?
2. If not relevant: What is the single most important missing data type to search for next?{{#unresolved_sub_queries}} If a sub-query has failed multiple times, consider rewriting it with different terminology. If many differently-worded attempts (e.g. 5) for the same goal have all failed, set data_not_available to true instead of continuing.{{/unresolved_sub_queries}}
3. Can the original query now be fully answered with the context and this candidate (if relevant)?