Query: {{query}}

Current cumulative result list preview:
```
{{result_list_preview}}
```
{{#is_continuation}}
This is a continuation: you previously evaluated completeness and decided to keep querying. The result list above is the updated deduplicated cumulative result. Decide afresh whether it is now complete.
{{/is_continuation}}

{{estimator_stats}}

Decide whether the result list is complete. If it is, call `stop()`. If more
agents are needed, call `continue_querying()`.
