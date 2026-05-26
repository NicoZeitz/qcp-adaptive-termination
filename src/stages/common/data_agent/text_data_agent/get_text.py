from functools import lru_cache

from qtypes.data_agent import DataAgent


@lru_cache(maxsize=None)
def get_text(data_agent: DataAgent) -> str:
    with open(data_agent.path) as f:
        return f.read()
