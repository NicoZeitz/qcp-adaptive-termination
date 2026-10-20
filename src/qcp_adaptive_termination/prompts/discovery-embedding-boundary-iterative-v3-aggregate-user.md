【Original query】
{{query}}{{#original_query_found_nothing}}

The original query as-is matched no relevant agents. Break it down into its most fundamental data building blocks, the atomic data types needed to answer it, and return each as a separate sub-query.{{/original_query_found_nothing}}

【Confirmed relevant agents so far】
{{agent_coverage_summary}}{{#pending_sub_queries}}

【Sub-queries already queued (do not repeat these)】
{{pending_sub_queries}}{{/pending_sub_queries}}{{#failed_sub_queries}}

【Failed sub-queries (found no new agents)】
{{failed_sub_queries}}{{/failed_sub_queries}}

Is the query fully answerable now? If not, what data types are still missing?
