from typing import Literal

from agents.run_config import DEFAULT_MAX_TURNS
from pydantic import BaseModel, ConfigDict
from openai.types.shared import Reasoning


class AgentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str | None
    reasoning: Reasoning
    verbosity: Literal["low", "medium", "high"] | None = None
    max_turns: int | None = DEFAULT_MAX_TURNS * 2
