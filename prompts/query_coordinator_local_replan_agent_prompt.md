The original user question is: "{{{query}}}"

A node in the query plan failed. Here is the exception — it contains the failing node (including which agent was asked, what sub-answers were available to it, and what it returned or requested):

```json
{{{exception}}}
```

Produce a corrected replacement for the failing node. Your output replaces exactly this node in the larger plan — keep the same `from_agent` and `need` as the original so it fits its parent.
