from dataclasses import dataclass
from typing import Literal
from agents import Model
from openai.types.shared import Reasoning


@dataclass
class AgentConfiguration:
    name: str
    model: Model | str | None
    reasoning: Reasoning
    verbosity: Literal["low", "medium", "high"] | None = None
