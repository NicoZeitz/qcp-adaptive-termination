from contextlib import contextmanager
from contextvars import ContextVar

from qtypes.multi_agent_iteration_record import MultiAgentIterationRecord


multi_agent_iteration_records: dict[str, list[MultiAgentIterationRecord]] = {}

current_multi_agent_iteration_records: ContextVar[str] = ContextVar(
    "current_multi_agent_iteration_records"
)


def add_multi_agent_iteration_record(
    record: MultiAgentIterationRecord,
):
    execution_id = current_multi_agent_iteration_records.get()

    if execution_id not in multi_agent_iteration_records:
        multi_agent_iteration_records[execution_id] = []

    multi_agent_iteration_records[execution_id].append(record)


def get_multi_agent_iteration_records(
    execution_id: str,
) -> list[MultiAgentIterationRecord]:
    return multi_agent_iteration_records.get(execution_id, [])


def get_current_multi_agent_iteration_records() -> list[MultiAgentIterationRecord]:
    execution_id = current_multi_agent_iteration_records.get()
    return get_multi_agent_iteration_records(execution_id)


@contextmanager
def multi_agent_iteration_record_context(execution_id: str):
    token = current_multi_agent_iteration_records.set(execution_id)
    try:
        yield
    finally:
        current_multi_agent_iteration_records.reset(token)
