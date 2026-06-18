Query: {{query}}

Current result list:
```
{{result_list_preview}}
```

Estimator statistics:
```
{{estimator_summary}}
```

Remaining agents not yet queried: {{remaining_agents}}

Decide whether to call `final_result(df)` or `query_more_agents(agents)`.
{{#remaining_agents}}
If no remaining agents are listed, you must call `final_result`.
{{/remaining_agents}}
