You are given a natural language query and a database schema. Determine if the query is relevant to the database. A query is relevant if any part of it could be answered using the database, even in a partial, indirect, or abstract way. Even loose or inferred connections count. Only return False if there is no connection between the query and the schema.

{{m_schema}}