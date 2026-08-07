from functools import lru_cache
import os

from sqlalchemy import Engine, create_engine, text

from qcp_adaptive_termination.model.agent.data_agent import DataAgent
from qcp_adaptive_termination.model.execution_config import ExecutionConfig


@lru_cache(maxsize=None)
def get_db_engine_for_path(resolved_path: str) -> Engine:
    engine = create_engine(f"sqlite:///file:{resolved_path}?mode=ro&uri=true")
    with engine.connect() as connection:
        connection.execute(text("PRAGMA temp_store=MEMORY;"))
    return engine


def get_db_engine(data_agent: DataAgent) -> Engine:
    prefix = ExecutionConfig.get().data_path_prefix
    resolved_path = os.path.abspath(os.path.join(prefix, data_agent.path))
    return get_db_engine_for_path(resolved_path)
