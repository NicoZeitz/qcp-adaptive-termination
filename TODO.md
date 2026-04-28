Split Generation
* Data Noise (TRANSLATE, RENAME_COLUMNS, CHANGE_DATA_TYPE, USE_DATA_ABBREVIATIONS, SYNONYMS)
* Redundancies (Multiple agents having (partially) overlapping data)

Data Agents
* Prompting better
  * Python code execution (can't see output of sql_query() without anything, should use print instead)

Orchestrator Agent
* Prompting better
  * Planning: Wrong dependencies (e.g. with person id's, no needs, but answer)
  * Replanning: Takes failed node and just adds new nodes to the plan somewhere else, just keeps the failed node the same
* Replanning: Ask all data agents for the information needs again for participation (that are not currently participating)

Result Agent
* Prompting better
* Result Alignment by Code Generation
* Final Answer Synthesis better