"""Regression tests for the estimator and completeness-evaluation fixes."""

import math

import pandas as pd

from qcp_adaptive_termination.estimator import DataItem, Estimator
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.stages.execution.multi_agent.completeness_evaluator import _truncate_history
from qcp_adaptive_termination.stages.execution.multi_agent.estimator_utils import dataframe_to_data_items
from scripts.completeness_evaluation.ground_truth import compute_ground_truth, truth_frame
from scripts.utils.sql import literal_predicate_sets, replace_literal_predicates


def _estimator(sources: list[list[str]]) -> Estimator:
    """Build an estimator from per-source entity names."""
    config = Config().completeness.estimator.model_copy(deep=True)
    config.bucket.enabled = False
    items = [[DataItem(name=n, value=1.0, source_id=f"s{i}") for n in names] for i, names in enumerate(sources)]
    estimator = Estimator([[item] for item in items[0]], config)
    for source in items[1:]:
        estimator.add_source(source)
    return estimator


def test_interval_does_not_collapse_without_doubletons():
    estimator = _estimator([["a", "b"], ["c"], ["d"], ["e"], ["f"], ["g"]])
    lower, upper = estimator.estimated_count_bounds()
    assert upper > estimator.sample_count()
    assert lower >= estimator.sample_count()


def test_interval_collapses_when_everything_is_seen_repeatedly():
    estimator = _estimator([["a", "b"], ["a", "b"], ["a", "b"]])
    assert estimator.estimated_count_bounds() == (2, 2)


def test_missing_values_are_not_part_of_the_value_population():
    frame = pd.DataFrame({"id": [1, 2, 3], "v": [10.0, None, 30.0]})
    items = dataframe_to_data_items(frame, ["id"], "v", "agent")
    assert [i.value for i in items] == [10.0, 30.0]
    config = Config().completeness.estimator.model_copy(deep=True)
    estimator = Estimator([[item] for item in items], config)
    assert math.isclose(estimator.sample_avg(), 20.0)


def test_ground_truth_prefers_rows_with_values():
    cache: dict[str, pd.DataFrame | None] = {
        "a": pd.DataFrame({"id": [1, 2], "v": [None, 5.0]}),
        "b": pd.DataFrame({"id": [1], "v": [7.0]}),
    }
    truth = compute_ground_truth(truth_frame(cache, ["id"], "v"), "v")
    assert truth.count == 2
    assert truth.value_count == 2
    assert truth.value_sum == 12.0
    assert truth.value_median == 6.0


def test_literal_predicates_are_rewritten():
    sql = "SELECT m.match FROM m WHERE m.team1 = 47762 OR m.team2 = 47762"
    assert literal_predicate_sets(sql) == [frozenset({"47762"}), frozenset({"47762"})]
    rewritten = replace_literal_predicates(sql, frozenset({"47762"}), [1, 2])
    assert rewritten.count("IN (1, 2)") == 2


def test_history_keeps_last_turns():
    history = [
        {"role": "user", "content": "1"},
        {"type": "reasoning"},
        {"role": "user", "content": "2"},
        {"type": "function_call"},
        {"role": "user", "content": "3"},
    ]
    assert _truncate_history(history, 2) == history[2:]  # ty: ignore[invalid-argument-type]
    assert _truncate_history(history, 5) == history  # ty: ignore[invalid-argument-type]
