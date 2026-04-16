from abc import ABC, abstractmethod
from dataclasses import dataclass

from modules.data import PartialQueryAnswer

class DataAgent(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent"""
        pass

    @abstractmethod
    async def get_data_and_query_capabilities(self) -> str:
        """Return a string that describes the agent's data and query capabilities in Natural Language"""
        pass

    @abstractmethod
    async def get_schema_for_query(self, query: str) -> str | None:
        """Return a string that describes the agent's data and query capabilities, such as the database schema."""
        pass

    @abstractmethod
    async def answer_query(self, query: str, additional_information: str) -> PartialQueryAnswer:
        """Answer the given query and return the answer as a string. If the query cannot be answered with the current information, return a string needed"""
        pass