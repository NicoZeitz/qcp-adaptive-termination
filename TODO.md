Make the code more modular
* Different stages (Discovery, Planning, Execution)
* Classes to inject (With coordinator reasoning, without, ...)
* Cache stages (write-back, read-only)
* File exporter for span data + remove span limits (OpenAI, https://github.com/openai/openai-agents-python/blob/656baf8ead8c970529d2c935acecac70ddc4fdc9/src/agents/tracing/processors.py#L33)
* Explore & add boundary from full schema injection of coordinator vs only nl descriptions

API Design
* Input multiple folders for data agents
* 1 Runner per database

Datasets
* Execute BIRD Benchmark

Test differences
* Multiple questions in one query (IMDB) -> new json files (query, sql_query, answer, relevant_agents)
* Multiple irrelevant databases (IMDB + databases from BIRD)
* Split imdb e.g. 10 horizontal if it works (IMDB)

Code Changes
* Replanning prompting force not to reuse step ids if not the same
* Replanning: Ask all data agents for the information needs again for participation (that are not currently participating)

Split Generation
* Data Noise (TRANSLATE, RENAME_COLUMNS, CHANGE_DATA_TYPE, USE_DATA_ABBREVIATIONS, SYNONYMS)
* Redundancies (Multiple agents having (partially) overlapping data)

```python
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

# 1. Create an in-memory SQLite database and load your DataFrame
mem_conn = sqlite3.connect(':memory:')
df.to_sql('b', mem_conn, index=False, if_exists='replace')
mem_conn.commit()

# 2. Get the in-memory database connection
# For in-memory SQLite with ATTACH, you need to use the same connection

# 3. Create your main SQLAlchemy session (assuming it's SQLite)
engine = create_engine('sqlite:///main.db')

# 4. Use raw SQL with ATTACH DATABASE
with engine.connect() as conn:
    # Attach the in-memory database
    conn.execute(text("ATTACH DATABASE ':memory:' AS secondary"))

    # Execute your cross-database query
    result = conn.execute(text("""
        SELECT a.name, b.title
        FROM main.a AS a
        INNER JOIN secondary.b AS b ON a.id = b.id
    """))

    for row in result:
        print(row)
```