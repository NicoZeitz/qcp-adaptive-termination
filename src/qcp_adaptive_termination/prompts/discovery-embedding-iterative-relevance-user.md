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
1. Is the candidate agents database actually relevant to the original query{{#matched_query}} or the search query that retrieved it{{/matched_query}}?
2. If not relevant: What sub-queries derived from the original query should be used to find the missing databases?{{#unresolved_sub_queries}} If a sub-query has failed multiple times, consider rewriting it with different terminology.{{/unresolved_sub_queries}}
3. Can the original query now be fully answered with the context and this candidate (if relevant)?