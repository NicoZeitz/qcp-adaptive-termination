import os

from sqlalchemy import Engine, create_engine, text

from qcp_adaptive_termination.models import Config, DataAgent

_engine_cache: dict[str, Engine] = {}


def _get_db_engine_for_path(resolved_path: str) -> Engine:
    if resolved_path not in _engine_cache:
        engine = create_engine(f"sqlite:///file:{resolved_path}?mode=ro&uri=true")
        with engine.connect() as connection:
            connection.execute(text("PRAGMA temp_store=MEMORY;"))
        _engine_cache[resolved_path] = engine
    return _engine_cache[resolved_path]


def get_db_engine(data_agent: DataAgent) -> Engine:
    prefix = Config.get().data_path_prefix
    resolved_path = os.path.abspath(os.path.join(prefix, data_agent.path))
    return _get_db_engine_for_path(resolved_path)


def quote_table(name: str) -> str:
    return ".".join(f'"{p}"' for p in name.split(".", 1))


def quote_col(name: str) -> str:
    return f'"{name}"'
