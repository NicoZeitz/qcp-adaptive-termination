


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
