from functools import lru_cache
import os

from sqlalchemy import Engine, create_engine, text

from qtypes.data_agent import DataAgent


@lru_cache(maxsize=None)
def get_db_engine(data_agent: DataAgent) -> Engine:
    path = os.path.abspath(data_agent.path)
    db_engine = create_engine(f"sqlite:///file:{path}?mode=ro&uri=true")
    with db_engine.connect() as connection:
        connection.execute(text("PRAGMA temp_store=MEMORY;"))
    return db_engine
