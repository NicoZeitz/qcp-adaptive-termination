Multiple source agents
agent round, pack other below it
check batch query estimation and iteration records
think more about query more prompt


Kill schema alignment step
* 1 at the start based on the output of the first agent
* the code agents have to do the rest by themselves

In Execution
* Check iteration records works with new stuff
* Check history how it works + implement own changes
* Multiple value columns possible
* Count + Sum query both possible

* Fix aggregation type by providing examples
* history of sample coverage changes
* always provide count statistics
* add number of new unique items found to estimator statistics (for latest agent)

Split Generation
* Data Noise (TRANSLATE, RENAME_COLUMNS, CHANGE_DATA_TYPE, USE_DATA_ABBREVIATIONS, SYNONYMS)
* Redundancies (Multiple agents having (partially) overlapping data)

```bash
curl -s dgx02.lab.tuda.systems:13505/version
{"version":"0.1.dev17492+g454b47db8"}

curl -s dgx02.lab.tuda.systems:13505/version -H "Content-Type: application/json" -d @curl.json | python3 -m json.tool
```