"""Build identical historical contexts and origin-masked OTB covariates for A/B/C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deeppace_net.foundation import FoundationConfig, calendar
from deeppace_net.pace_data import PaceData

COMPACT_GRID = (1, 3, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)


@dataclass(frozen=True)
class CovariateFrames:
    history: pd.DataFrame
    future: pd.DataFrame
    history_days: int
    observed_history_days: int


def build_covariates(
    data: PaceData,
    property_id: int,
    as_of: str | pd.Timestamp,
    horizon: int,
    config: FoundationConfig,
    case: str = "A",
    grid: tuple[int, ...] = COMPACT_GRID,
) -> CovariateFrames:
    """C is a separate h-step task: current_otb means OTB at DTA h in every row.

    Only the endpoint forecast is scored. Historical completed curves are legitimately
    available at t. For future arrival t+s, a DTA k cell is available iff k >= s.
    Unknown cells are NaN, never zero, and DTA 0 is never a covariate.
    """
    if case not in {"A", "B", "C"} or not 1 <= horizon <= data.k:
        raise ValueError("Invalid case or horizon")
    if case == "C" and (not grid or min(grid) < 1 or max(grid) > data.k):
        raise ValueError(f"Compact DTA grid must be within 1..{data.k}")
    t = pd.Timestamp(as_of)
    hist = data.history[data.history.property_id == property_id].set_index("stay_date")
    dates = pd.date_range(end=t, periods=config.context_days)
    target = hist.rooms_sold.reindex(dates)
    if target.notna().any():
        target = target.loc[target.first_valid_index() :]
    if target.notna().sum() < config.min_history or pd.isna(target.iloc[-1]):
        raise ValueError(f"Insufficient history for property {property_id} at {t.date()}")
    history = calendar(pd.DatetimeIndex(target.index))
    history["target"] = target.to_numpy()
    future = calendar(pd.date_range(t + pd.Timedelta(days=1), periods=horizon))
    for frame in (history, future):
        frame["item_id"] = str(property_id)

    def column(frame: pd.DataFrame, k: int) -> np.ndarray:
        dates = pd.DatetimeIndex(frame.timestamp)
        known = dates - pd.Timedelta(days=k) <= t
        values = np.full(len(dates), np.nan, dtype=np.float32)
        keys = pd.MultiIndex.from_tuples([(property_id, d, k) for d in dates[known]])
        values[known] = data.surface.reindex(keys).to_numpy(dtype=np.float32)
        # Missing source observations remain missing rather than silently filled.
        return values

    selected = range(1, data.k + 1) if case == "B" else sorted(set(grid)) if case == "C" else []
    for frame_name, frame in (("history", history), ("future", future)):
        columns = {f"otb_dta_{k}": column(frame, k) for k in selected}
        if case == "C":
            columns["current_otb"] = column(frame, horizon)
        if columns:
            combined = pd.concat([frame, pd.DataFrame(columns)], axis=1)
            if frame_name == "history":
                history = combined
            else:
                future = combined
    return CovariateFrames(history, future, len(target), int(target.notna().sum()))
