The Query Coordinator thinks you might be relevant for this query. Please evaluate whether you agree.{{coordinator_reasoning}}

【Question】
{{query}}

**Your Task:**
Evaluate whether you can actually help answer this question with your text data.

You must provide TWO things:
1. **is_relevant** (bool): True if you can help, False if you cannot
2. **reasoning** (string): A clear explanation of your decision

**Return True only if:**
- Your text content contains relevant information
- The query can be answered (fully or partially) by analyzing your text
- You have reasonable confidence you can provide a useful answer

**If you return False, explain:**
- What specific information is missing from your text
- Why the coordinator's reasoning doesn't match your actual content
- What would be needed to answer this query (if you know)

Be honest and conservative. It's better to decline than to accept a query you can't properly handle.