The Coordinator thinks you might be relevant for this question.

【Question】
{{query}}

【Coordinator】
{{coordinator_reasoning}}

**Your Task:**
Evaluate whether you can actually help answer this question with your database schema.

You must provide TWO things:
1. **is_relevant** (bool): True if you can help, False if you cannot
2. **reasoning** (string): A clear explanation of your decision

**Return True only if:**
- Your schema contains the specific data the coordinator asks for
- You can retrieve the information (fully or partially) using SQL queries
- You have reasonable confidence you can help with the task

**If you return False, explain:**
- What specific data or capability is missing from your schema
- Why the coordinator's reasoning doesn't match your actual capabilities
- What would be needed to answer this query (if you know)

Be honest and conservative. It's better to decline a query you can't properly handle.