from agents import Agent
import os
import time

from modules.model import AgentContext
from modules.splitter import SplitType, create_agents

if __name__ == "__main__":
    always_included_columns_mapping = {
        # pk mappings
        "titles": ["title_id"],
        "ratings": ["title_id"],
        "people": ["person_id"],

        # needed
        "akas": ["title_id"],
        "crew": ["title_id", "ordering", "person_id"],
        "episodes": ["episode_title_id"],
    }

    start_time = time.time()

    data_agents = create_agents(
        output_path="./data/splits",
        database_url=f'sqlite:///{os.path.abspath(f"./data/imdb.db")}',
        agent=Agent[AgentContext](name=""),
        agent_amount=5,
        always_included_columns_mapping=always_included_columns_mapping,
        split_type=SplitType.MIXED,
        force_splits=False,
        max_splits_per_table=3,
        seed=42
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")