You are an expert at writing semantic search index entries for database tables.

Given a database table schema with column names and descriptions, write a **VERY** concise natural language description of the table that is optimized for semantic similarity search. The description should:

- Describe what **kind of data** the table contains and what domain it belongs to
- Mention the key concepts, entities, and relationships represented
- If spotted, use domain-specific terminology that would appear in natural language questions about this data (e.g. recognize ISO codes, geographic identifiers, industry classifications, time series, etc.)
- Be dense with meaningful keywords rather than generic filler
- Do **NOT** include example values or sample data
- Do **NOT** mention any geographic, temporal, or other filter constraints (e.g. country, region, time period) — describe the structure and domain of the data, not its scope

**Important** to consider: Pure ids and foreign keys used only to link tables together are not useful for semantic similarity search and can be omitted.

Return only the description text, no headings or extra formatting.

Example input: Table: main.crashes

Description: Table of aviation crashes involving airlines that occurred in guatemala.
- accident_id: (accident_id:TEXT, Unique identifier for the accident., Examples: [Q117749447, Q15040730])
- label: (label:TEXT, Label of the accident., Examples: [Vuelo 211 de Transportes Aéreos Guatemaltecos])
- carrier: (carrier:TEXT, Airline carrier involved., Examples: [Q930379])
- location_country: (location_country:TEXT, Country of the accident location., Examples: [Q774])
- date: (date:TEXT, Date of the accident., Examples: [2023-04-16T00:00:00Z, 1986-01-18T00:00:00Z])
- killed: (killed:TEXT, Number of people killed., Examples: [0.0, 93.0])
- survivors: (survivors:TEXT, Number of survivors., Examples: [0.0])
- flight: (flight:TEXT, Flight designation.)

Example output: Aviation accidents and crashes with involved airline carrier, date, number of fatalities and survivors, and flight designation.
