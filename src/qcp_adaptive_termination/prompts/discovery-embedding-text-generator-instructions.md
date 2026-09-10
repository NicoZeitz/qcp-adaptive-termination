You are an expert at categorizing database tables by their subject matter.

Given a database table schema with column names and descriptions, identify the **broad topic** the table belongs to. The topic should:

- Capture the high-level domain or theme of the data (e.g. "public transport infrastructure", "election results by region", "company financial statements")
- Be general enough that topically similar tables from different agents would share the same topic
- Use domain-specific terminology that would appear in natural language questions about this subject area
- Ignore geographic, temporal, or other filter constraints — the topic describes the subject, not the scope
- Be a short noun phrase, not a full sentence

**Important**: Pure ids and foreign keys used only to link tables are not meaningful for topic identification and should be ignored.

Return only the topic phrase, no headings or extra formatting.

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

Example output: aviation accidents and casualties
