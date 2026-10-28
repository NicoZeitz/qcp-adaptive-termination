"""Ground-truth computation for completeness evaluation."""

import numpy as np
import pandas as pd

from scripts.completeness_evaluation.models import GroundTruth


def truth_frame(
    data_cache: dict[str, pd.DataFrame | None],
    key_cols: list[str],
    value_column: str | None = None,
) -> pd.DataFrame:
    """Deduplicated union of all agents' rows, preferring rows with a value per key."""
    frames = [df for df in data_cache.values() if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if value_column is not None and value_column in combined.columns:
        missing = np.isnan(np.asarray(pd.to_numeric(combined[value_column], errors="coerce"), dtype=float))
        combined = combined.iloc[np.argsort(missing, kind="stable")]
    dedup_cols = [c for c in key_cols if c in combined.columns] or list(combined.columns)
    return combined.drop_duplicates(subset=dedup_cols)


def truth_keys(deduped: pd.DataFrame, key_cols: list[str]) -> set[tuple]:
    """Key tuples of a deduplicated truth frame."""
    cols = [c for c in key_cols if c in deduped.columns]
    if deduped.empty or not cols:
        return set()
    return set(map(tuple, deduped[cols].values.tolist()))


def compute_ground_truth(deduped: pd.DataFrame, value_column: str | None) -> GroundTruth:
    """Compute ground truth (distinct count and value stats) from a deduplicated truth frame."""
    value_sum = value_avg = value_min = value_max = value_median = None
    value_count = 0
    if value_column is not None and value_column in deduped.columns:
        values = np.asarray(pd.to_numeric(deduped[value_column], errors="coerce"), dtype=float)
        values = values[~np.isnan(values)]
        value_count = len(values)
        if value_count:
            value_sum = float(values.sum())
            value_avg = float(values.mean())
            value_min = float(values.min())
            value_max = float(values.max())
            value_median = float(np.median(values))

    return GroundTruth(
        count=len(deduped),
        value_count=value_count,
        value_sum=value_sum,
        value_avg=value_avg,
        value_min=value_min,
        value_max=value_max,
        value_median=value_median,
    )
