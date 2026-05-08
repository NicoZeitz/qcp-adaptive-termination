from data_agent.data_agent import DataAgent
from data_agent.text_data_agent.answer_query import answer_query
from data_agent.text_data_agent.is_relevant import is_relevant
from data_agent.text_data_agent.data_and_nl_description import (
    data_description,
    natural_language_description,
)
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.partial_query_answer import PartialQueryAnswer


class TextDataAgent(DataAgent):
    def __init__(self, name: str, text: str):
        super().__init__(name)
        self.text = text

    async def data_description(self) -> str:
        return await data_description(self, self.text)

    async def natural_language_description(self) -> str:
        return await natural_language_description(self, self.text)

    async def query_capabilities(self) -> list[str]:
        return ["Natural Language Question Answering", "Reading Comprehension"]

    async def is_relevant(self, query: str, reasoning: str) -> bool:
        return await is_relevant(self, self.text, query, reasoning)

    async def answer_query(
        self, query: str, additional_information: list[AdditionalInformation]
    ) -> PartialQueryAnswer:
        return await answer_query(self, self.text, query, additional_information)
