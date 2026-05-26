from functools import lru_cache
import os

from sqlalchemy import Engine, create_engine

from qtypes.data_agent import DataAgent


@lru_cache(maxsize=None)
def get_db_engine(data_agent: DataAgent) -> Engine:
    return create_engine(f"sqlite:///{os.path.abspath(data_agent.path)}")
