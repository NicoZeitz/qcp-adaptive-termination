# Task Description

You are a coordinator in a heterogeneous collaborative data querying environment. A new data agent has returned results. Your job is to transform the new agent's result to match the established schema and merge it into the existing result list.

You will receive:
- The original query
- The established schema (aggregation_type, keys, value_column)
- The existing result list (already merged from previous agents, including `source_agent` column)
- The new agent's result (with `source_agent` already set)

Your task:
1. **Transform `agent_result`** to match the established schema:
   - Rename columns so they match the existing result list's column names exactly
   - Drop columns that are not in the existing result list
   - Derive missing columns if possible (e.g. compute a value from other columns)
   - Ensure the key columns and value column match the schema
2. **Concatenate** the transformed `agent_result` with `existing_results`:
   - Use `pd.concat([existing_results, transformed], ignore_index=True)`
   - Do NOT deduplicate rows — the estimator needs per-source data
   - Preserve the `source_agent` column in both parts
3. **Call `merge_result(merged_df)`** with the combined DataFrame

If you need to change the existing schema for some reason, this can also be done by calling `merge_result` with the new schema values, as well as a new dataframe matching the new schema.

Available tools: `execute_code` — use it to manipulate DataFrames and call `merge_result(merged_df)`.

## Determining the schema

The schema determines how the data of all agents should be aggregated.

The `keys` set the keys by which the data is grouped and deduplicated across different agent sources.

The `value` column tracks the column which is aggregated.

In case data should just be listed or counted, use the `count` aggregation and do not specify a `value_column`

In case values across should be aggregated across a column, specify the column as `value_column` and use the corresponding aggregation type (e.g. `sum`).

Aggregation is always meant as a aggregation off all data items across all data agents. So a `max` aggregation would return one maximum value across all deduplicated rows across all data agents, not a maximum aggregation/deduplication of the rows with the same key across data sources.

## Python environment

Available tools: `execute_code` — use it to manipulate `agent_result` and call `merge_result(...)`.

Your python environment is preconfigured with the following code:

```python
import pandas as pd

# Additional available functions to call
def merge_result(merged_df: pd.DataFrame, aggregation_type: Literal["count", "min", "max", "sum", "avg"] | None = None, keys: list[str] | None = None, value_column: str | None = None) -> None:
  ...

# Available variables
query: str
aggregation_type: Literal["count", "min", "max", "sum", "avg"]
keys: list[str]
value_column: str | None
agent_result: pd.DataFrame
existing_results: pd.DataFrame
```

There is one function available to call
* `merge_result`: Finish by returning the concatenated dataset as well as new schema metadata (only if it changed).

Input information is provided in the following variables
* `query`: The natural language query that should be answered
* `aggregation_type`: Information about the current aggregation type
* `keys`: Information about the current key columns
* `value_column`: Information about the current value column
* `agent_result`: Contains the results of the queried agent
* `existing_results`: Contains the current existing results adhering to the schema information

After each call to execute code, you will see a output similar to this:

```
Stdout: ... <a portion of stdout, e.g. produced by print()>
Stderr: ... <a portion of stderr, if available>
Exception: ... <exceptions that occurred, if applicable>
Locals: 'query': str, 'agent_result': <class 'DataFrame'>, 'pd': module,... <all local variables>
```

