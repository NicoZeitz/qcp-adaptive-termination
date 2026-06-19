from qtypes.aligned_result_list import AlignedResultList
from qtypes.result import ResultType
from stages.execution.answer_step_multiple_agents.align_schema_continue import (
    align_schema_continue,
)
from stages.execution.answer_step_multiple_agents.align_schema_first import (
    align_schema_first,
)


async def align_schema(
    aligned: AlignedResultList,
    agent_name: str,
    agent_result: ResultType,
    query: str,
) -> AlignedResultList:
    if aligned.schema_metadata is None:
        return await align_schema_first(query, agent_name, agent_result)
    else:
        return await align_schema_continue(query, agent_name, agent_result, aligned)
