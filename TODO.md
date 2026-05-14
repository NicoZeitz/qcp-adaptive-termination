Make the code more modular
* Different stages (Discovery, Planning, Execution)
* Classes to inject (With coordinator reasoning, without, ...)
* Cache stages (write-back, read-only)
* File exporter for span data + remove span limits (OpenAI, https://github.com/openai/openai-agents-python/blob/656baf8ead8c970529d2c935acecac70ddc4fdc9/src/agents/tracing/processors.py#L33)

API Design
* Input multiple folders for data agents
* 1 Runner per database

Datasets
* Execute BRID Benchmark

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