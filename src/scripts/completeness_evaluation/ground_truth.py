"""Ground-truth computation for completeness evaluation."""

from typing import Any, cast

import pandas as pd

from scripts.completeness_evaluation.models import GroundTruth


def compute_ground_truth(
    data_cache: dict[str, pd.DataFrame | None],
    key_cols: list[str],
    value_column: str | None,
) -> GroundTruth:
    frames = [df for df in data_cache.values() if df is not None and not df.empty]
    if not frames:
        return GroundTruth(count=0, value_sum=None, value_avg=None, value_min=None, value_max=None)

    combined = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in key_cols if c in combined.columns] or list(combined.columns)
    deduped = combined.drop_duplicates(subset=dedup_cols)

    value_sum = value_avg = value_min = value_max = None
    if value_column is not None and value_column in deduped.columns:
        col = deduped[value_column].dropna()
        if not col.empty:
            value_sum = float(cast(Any, col.sum()))
            value_avg = float(cast(Any, col.mean()))
            value_min = float(cast(Any, col.min()))
            value_max = float(cast(Any, col.max()))

    return GroundTruth(
        count=len(deduped),
        value_sum=value_sum,
        value_avg=value_avg,
        value_min=value_min,
        value_max=value_max,
    )
