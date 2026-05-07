For each agent in the registry, determine:
1. **is_relevant**: Whether the agent might be able to help answer the query (even partially or indirectly)
2. **reasoning**: A clear explanation of WHY you think the agent is or isn't relevant

The reasoning will be shared with the agent so they can evaluate whether they agree with your assessment. Be specific about:
- What data/capabilities the agent has that relate to the query
- How the agent could contribute to answering the query
- What assumptions you're making about the agent's usefulness

An agent is potentially relevant if it could provide information that helps answer the question, even if only partially. The agent is relevant if it could help in any indirect, partial, or abstract way. Only mark as not relevant if there is absolutely no connection between the agent's capabilities and the question.

【Agent Registry】
A3: 【DB_ID】 A3
【Schema】
# Table: main.episodes
[
(episode_title_id:VARCHAR, Unique identifier for the specific episode title., Examples: [tt0031458, tt0041951, tt0042816]),
(show_title_id:VARCHAR, Identifier of the parent TV series to which the episode belongs., Examples: [tt32857063, tt0041038, tt0989125]),
(season_number:INTEGER, Season of the series in which the episode appears, numbered sequentially., Examples: [1, 3, 2]),
(episode_number:INTEGER, Order of the episode within its season, starting from 1., Examples: [9, 17, 42])
]
# Table: main.ratings
[
(title_id:VARCHAR, Identifier of the title (e.g., a series, episode, or other work) being rated., Examples: [tt0000001, tt0000002, tt0000003]),
(rating:REAL, Average user rating for the title, typically on a 0–10 scale with decimals., Examples: [5.7, 5.5, 6.4]),
(votes:INTEGER, Total number of user ratings submitted for the title., Examples: [2205, 313, 2318])
]
A2: 【DB_ID】 A2
【Schema】
# Table: main.crew
[
(title_id:VARCHAR, Unique identifier of the film or TV title this credit belongs to., Examples: [tt0000001, tt0000002, tt0000003]),
(ordering:INTEGER, Numeric position of this credit in the title's official listing (used to preserve display order)., Examples: [1, 2, 3]),
(person_id:VARCHAR, Unique identifier of the credited person (actor/performer or crew member)., Examples: [nm1588970, nm0005690, nm0374658]),
(category:VARCHAR, Broad credit category for the person on this title (e.g., director, producer, self/performer)., Examples: [self, director, producer]),
(job:VARCHAR, Specific job title or role description within the category; may be blank if redundant with the category., Examples: [producer]),
(characters:VARCHAR, Character name or names portrayed when the credit involves on‑screen appearance; otherwise empty or not applicable., Examples: [["Self"], ["Blacksmith"], ["Assistant"]])
]
A5: 【DB_ID】 A5
【Schema】
# Table: main.akas
[
(title_id:VARCHAR, Unique identifier of the title this alternate name belongs to; corresponds to main.titles.title_id., Examples: [tt0000001, tt0000002, tt0000003]),
(title:VARCHAR, The alternate or localized title text as used in the specified region/language., Examples: [Carmencita]),
(region:VARCHAR, Region or country code where this title variant is used (e.g., ISO 3166-1)., Examples: [DE, US, HU]),
(language:VARCHAR, Language code of the title text (e.g., ISO 639-1)., Examples: [ja, en, sv]),
(types:VARCHAR, Category tags describing the title variant (e.g., original, IMDb display, alternative); may contain multiple values., Examples: [original, imdbDisplay, alternative]),
(attributes:VARCHAR, Additional qualifiers about the title variant (e.g., literal title, working title)., Examples: [literal title]),
(is_original_title:INTEGER, Flag indicating whether this is the original title (1) or not (0)., Examples: [1, 0])
]
# Table: main.titles
[
(title_id:VARCHAR, Unique identifier for the title (commonly an IMDb-style tconst)., Examples: [tt2617794, tt26177948, tt2617796]),
(type:VARCHAR, The work's category or format (e.g., movie, short, tvEpisode)., Examples: [tvEpisode, movie, short]),
(primary_title:VARCHAR, The canonical display title used for general reference., Examples: [Families, Episode #1.13, Ms. Phillips]),
(original_title:VARCHAR, The title in the work's original language/script., Examples: [Families, Episode #1.13, Ms. Phillips]),
(is_adult:INTEGER, Flag indicating adult content: 1 for adult, 0 otherwise., Examples: [0, 1]),
(premiered:INTEGER, Year the title was first released or a series first aired., Examples: [1991, 1980, 2013]),
(ended:INTEGER, Year the title or series ended/last aired; empty if not applicable., Examples: [2011, 2013, 2014]),
(runtime_minutes:INTEGER, Approximate running time in minutes., Examples: [53, 6, 47]),
(genres:VARCHAR, Comma-separated list of genre labels associated with the title., Examples: [Crime,Drama])
]
A1: 【DB_ID】 A1
【Schema】
# Table: main.people
[
(person_id:VARCHAR, Unique string identifier for each person within the database., Examples: [nm0000001, nm0000002, nm0000003]),
(name:VARCHAR, The person's full name., Examples: [Fred Astaire, Lauren Bacall, Brigitte Bardot]),
(born:INTEGER, Four-digit year the person was born., Examples: [1899, 1924, 1934]),
(died:INTEGER, Four-digit year the person died; left blank if still living or unknown., Examples: [1987, 2014, 2025])
]
A4: 【DB_ID】 A4
【Schema】
# Table: main.titles
[
(title_id:VARCHAR, Unique stable identifier for the title within the dataset (e.g., an IMDb-style code)., Examples: [tt0000001, tt0000002, tt0000003]),
(type:VARCHAR, Category or format of the title, such as feature film, short, or TV short., Examples: [short, movie, tvShort]),
(primary_title:VARCHAR, The main title by which the work is commonly known in the primary market of the dataset., Examples: [Carmencita]),
(original_title:VARCHAR, The title as originally released in its original language or market., Examples: [Carmencita]),
(is_adult:INTEGER, Flag indicating adult/explicit content (1) versus non-adult/general content (0)., Examples: [0, 1]),
(premiered:INTEGER, Year the title was first released or first aired., Examples: [1894, 1892, 1893]),
(ended:INTEGER, Year the title or series concluded; not applicable for one-off releases or ongoing works., Examples: [1947, 1945, 1955]),
(runtime_minutes:INTEGER, Running time in minutes; for series, typically the per-episode length., Examples: [1, 5, 12]),
(genres:VARCHAR, Comma-separated list of genres associated with the title., Examples: [Documentary,Short])
]
