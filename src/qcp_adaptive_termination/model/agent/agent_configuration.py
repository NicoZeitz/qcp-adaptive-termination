from typing import Literal

from agents.run_config import DEFAULT_MAX_TURNS
from pydantic import BaseModel, ConfigDict
from openai.types.shared import Reasoning


class AgentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    agent_name: str | None = None
    model: str | None
    reasoning: Reasoning
    verbosity: Literal["low", "medium", "high"] | None = None
    max_turns: int | None = DEFAULT_MAX_TURNS * 2

    @property
    def display_name(self) -> str:
        return f"{self.agent_name} - {self.role}" if self.agent_name else self.role
