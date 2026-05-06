from abc import ABC, abstractmethod

from openai.types.shared import Reasoning

from qtypes.agent_configuration import AgentConfiguration
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan.result import ResultType


class DataAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    def agent_configuration(self) -> AgentConfiguration:
        return AgentConfiguration(
            name=self.name,
            model="gpt-5",
            reasoning=Reasoning(effort="high"),
        )

    @abstractmethod
    async def data_description(self) -> str:
        pass

    @abstractmethod
    async def query_capabilities(self) -> list[str]:
        pass

    @abstractmethod
    async def is_relevant(self, query: str) -> bool:
        pass

    @abstractmethod
    async def answer_query(
        self, query: str, additional_information: dict[str, ResultType]
    ) -> PartialQueryAnswer:
        pass
