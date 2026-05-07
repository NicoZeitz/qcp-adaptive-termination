from sqlalchemy import Engine

from data_agent.data_agent import DataAgent
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.partial_query_answer import PartialQueryAnswer
from data_agent.sql_data_agent.answer_query import answer_query
from data_agent.sql_data_agent.data_description import data_description
from data_agent.sql_data_agent.is_relevant import is_relevant


class SqlDataAgent(DataAgent):
    def __init__(self, name: str, db_engine: Engine):
        super().__init__(name)
        self.db_engine = db_engine

    async def data_description(self) -> str:
        return await data_description(self, self.db_engine)

    async def query_capabilities(self) -> list[str]:
        return ["Text-to-SQL"]

    async def is_relevant(self, query: str, reasoning: str) -> bool:
        return await is_relevant(self, query, reasoning)

    async def answer_query(
        self, query: str, additional_information: list[AdditionalInformation]
    ) -> PartialQueryAnswer:
        return await answer_query(self, self.db_engine, query, additional_information)
