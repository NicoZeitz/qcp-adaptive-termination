from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from qtypes.estimator_result import EstimatorResult

EvaluatorDecision = Literal["final_result", "query_more_agents"]


@dataclass
class MultiAgentIterationRecord:
    step_id: str
    agent_name: str
    result_list: pd.DataFrame
    estimator_result: EstimatorResult | None
    evaluator_decision: EvaluatorDecision
    evaluator_next_agents: list[str] = field(default_factory=list)
