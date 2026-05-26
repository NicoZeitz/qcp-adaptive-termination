from functools import lru_cache
import os

from sqlalchemy import Engine, create_engine, text

from qtypes.data_agent import DataAgent


@lru_cache(maxsize=None)
def get_db_engine(data_agent: DataAgent) -> Engine:
    db_engine = create_engine(f"sqlite:///{os.path.abspath(data_agent.path)}")
    with db_engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA foreign_keys=OFF;"))
        conn.execute(text("PRAGMA journal_mode=MEMORY;"))
        conn.execute(text("PRAGMA temp_store=MEMORY;"))
        conn.execute(text("PRAGMA synchronous=OFF;"))
        conn.execute(text("PRAGMA query_only = ON;"))
        conn.commit()

    return db_engine
