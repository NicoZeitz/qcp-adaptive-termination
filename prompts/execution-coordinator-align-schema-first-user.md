Query: {{query}}

First agent queried: {{agent_name}}

Agent result preview:
```
{{agent_result_preview}}
```

Transform `agent_result` to best answer the query (rename, drop, or derive columns), then call `set_schema(result_df, aggregation_type, keys, value_column)` with the transformed DataFrame.
