"""
Data agent loading and database engine setup.

Loads agent registry data from disk and constructs SQLAlchemy engines
for the databases each agent is responsible for.
"""

from qcp_adaptive_termination.data_agent.data_agent import load_agent_registry_data
from qcp_adaptive_termination.db.get_db_engine import get_db_engine

__all__ = ["load_agent_registry_data", "get_db_engine"]
