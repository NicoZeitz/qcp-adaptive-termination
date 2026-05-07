import textwrap

from async_lru import alru_cache
from pydantic import BaseModel, create_model, Field
from sqlalchemy import Engine

from data_agent.data_agent import DataAgent
from data_agent.sql_data_agent.mschema.schema_engine import SchemaEngine
from utils.prompt import load_prompt
from utils.run_agent import run_agent

CACHED_DATA_DESCRIPTION = {
    "A1": textwrap.dedent("""
        【DB_ID】 A1
        【Schema】
        # Table: main.people
        [
        (person_id:VARCHAR, Unique string identifier for each person within the database., Examples: [nm0000001, nm0000002, nm0000003]),
        (name:VARCHAR, The person’s full name., Examples: [Fred Astaire, Lauren Bacall, Brigitte Bardot]),
        (born:INTEGER, Four-digit year the person was born., Examples: [1899, 1924, 1934]),
        (died:INTEGER, Four-digit year the person died; left blank if still living or unknown., Examples: [1987, 2014, 2025])
        ]
        """).strip(),
    "A2": textwrap.dedent("""
        【DB_ID】 A2
        【Schema】
        # Table: main.crew
        [
        (title_id:VARCHAR, Unique identifier of the film or TV title this credit belongs to., Examples: [tt0000001, tt0000002, tt0000003]),
        (ordering:INTEGER, Numeric position of this credit in the title’s official listing (used to preserve display order)., Examples: [1, 2, 3]),
        (person_id:VARCHAR, Unique identifier of the credited person (actor/performer or crew member)., Examples: [nm1588970, nm0005690, nm0374658]),
        (category:VARCHAR, Broad credit category for the person on this title (e.g., director, producer, self/performer)., Examples: [self, director, producer]),
        (job:VARCHAR, Specific job title or role description within the category; may be blank if redundant with the category., Examples: [producer]),
        (characters:VARCHAR, Character name or names portrayed when the credit involves on‑screen appearance; otherwise empty or not applicable., Examples: [["Self"], ["Blacksmith"], ["Assistant"]])
        ]
        """).strip(),
    "A3": textwrap.dedent("""
        【DB_ID】 A3
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
        """).strip(),
    "A4": textwrap.dedent("""
        【DB_ID】 A4
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
        """).strip(),
    "A5": textwrap.dedent("""
        【DB_ID】 A5
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
        (type:VARCHAR, The work’s category or format (e.g., movie, short, tvEpisode)., Examples: [tvEpisode, movie, short]),
        (primary_title:VARCHAR, The canonical display title used for general reference., Examples: [Families, Episode #1.13, Ms. Phillips]),
        (original_title:VARCHAR, The title in the work’s original language/script., Examples: [Families, Episode #1.13, Ms. Phillips]),
        (is_adult:INTEGER, Flag indicating adult content: 1 for adult, 0 otherwise., Examples: [0, 1]),
        (premiered:INTEGER, Year the title was first released or a series first aired., Examples: [1991, 1980, 2013]),
        (ended:INTEGER, Year the title or series ended/last aired; empty if not applicable., Examples: [2011, 2013, 2014]),
        (runtime_minutes:INTEGER, Approximate running time in minutes., Examples: [53, 6, 47]),
        (genres:VARCHAR, Comma-separated list of genre labels associated with the title., Examples: [Crime,Drama])
        ]
        """).strip(),
}


def create_database_description_model(schema_engine: SchemaEngine) -> type[BaseModel]:
    schema_dict = {}
    for table_name, table_info in schema_engine.mschema.tables.items():
        fields = {}
        for col_name in table_info["fields"].keys():
            fields[col_name] = (str, Field(...))
        schema_dict[table_name] = (
            create_model(f"{table_name}ColumnDescriptions", **fields),
            Field(...),
        )
    return create_model("DatabaseDescriptionModel", **schema_dict)


@alru_cache
async def data_description(data_agent: DataAgent, db_engine: Engine) -> str:
    if data_agent.name in CACHED_DATA_DESCRIPTION:
        print("Using cached data description for agent:", data_agent.name)
        return CACHED_DATA_DESCRIPTION[data_agent.name]  # TODO:REMOVE

    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)
    m_schema_without_comments = schema_engine.mschema.to_mschema()

    DatabaseDescriptionModel = create_database_description_model(schema_engine)

    result, _ = await run_agent(
        name_postfix="Schema Description",
        agent_configuration=data_agent.agent_configuration,
        instructions=load_prompt("SQL Data Agent/Schema Description/Instructions", {}),
        prompt=load_prompt(
            "SQL Data Agent/Schema Description/Prompt",
            {"m_schema_without_comments": m_schema_without_comments},
        ),
        output_type=DatabaseDescriptionModel,
    )

    # Update the schema with the generated descriptions
    for table_name, table_info in schema_engine.mschema.tables.items():
        for col_name, field_info in table_info["fields"].items():
            description = getattr(getattr(result, table_name, {}), col_name, None)
            if description:
                field_info["comment"] = description

    return schema_engine.mschema.to_mschema()
