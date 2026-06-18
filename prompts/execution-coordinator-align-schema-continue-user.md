Query: {{query}}

Schema: `{{schema}}`

Existing result list columns: `{{existing_result_columns}}`
Existing result list preview:
```
{{existing_result_preview}}
```

New agent: **{{agent_name}}**
New agent result columns: `{{agent_result_columns}}`
New agent result preview:
```
{{agent_result_preview}}
```

Transform `agent_result` to match the existing schema (rename/drop/derive columns as needed), concatenate with `existing_result_list`, and call `merge_result(merged_df)`.
