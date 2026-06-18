Problems with execution
* No custom spans for subcalls (schema alignement, completeness eval)
* Evaluators weird

Other
* Rename stages (Discovery, Planning, Execution)

Scaling Tests
* Multiple questions in one query (IMDB) -> new json files (query, sql_query, answer, relevant_agents)
* Multiple irrelevant databases (IMDB + databases from BIRD)
* Split imdb e.g. 10 horizontal if it works (IMDB)

Split Generation
* Data Noise (TRANSLATE, RENAME_COLUMNS, CHANGE_DATA_TYPE, USE_DATA_ABBREVIATIONS, SYNONYMS)
* Redundancies (Multiple agents having (partially) overlapping data)
