{{query}}

## Initial Query Plan (to be replanned)

The plan failed and needs to be replanned. Please create a new plan that addresses the failure reason below.

```json
{{old_query_plan}}
```

## Reason for Replanning

```json
{{reason}}
```

{{#is_continuation}}
This is a continuation: you previously drafted this query plan and it failed to execute. The reason above explains the failure. Amend the plan based on the full conversation history instead of starting from scratch.
{{/is_continuation}}
