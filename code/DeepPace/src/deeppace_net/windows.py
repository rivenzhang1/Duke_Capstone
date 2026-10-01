"""As-of windowing: slice the pace surface into training examples.

Design: docs/design.md §4.2, §5.

A training example is a 2D patch of the surface — arrival dates on one axis, DTA on the
other — sliced at an as-of date. Because the slice is diagonal in (arrival_date, dta) space,
the observed region is a **staircase, not a rectangle**: near arrival dates are observed deep
into small DTA, far ones only at large DTA. Masking that region is part of the model
contract, not a preprocessing detail.

**The leakage rule is absolute.** A feature for as-of `t` may use only booking activity with
``booking_date <= t``. The simulated parquet contains the complete curve for every arrival
date *including the future*, so this is easy to violate by accident.

**Slice on `booking_date`, never on DTA alone.** For arrival date `d` and as-of `t` the
observable region is ``dta >= (d - t)``. A fixed DTA cutoff applied across a window of
arrival dates leaks, because the same DTA is a different booking date for each `d`.

This invariant is covered by ``tests/test_no_leakage.py``, which fails if any cell with
``booking_date > as_of`` reaches the model.

Public surface:
    SurfaceWindow (dataclass) — values, observed_mask, arrival_dates, dta_grid, as_of, target
    build_window(on_books, as_of, arrival_dates, dta_grid, value_col, target) -> SurfaceWindow
    iter_training_windows(on_books, arrival_dates, dta_grid, as_of_dates, value_col, target)
        -> Iterator[SurfaceWindow]
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurfaceWindow:
    """One (arrival_date, DTA) patch of the surface, sliced at `as_of`.

    `values` is zero-filled outside `observed_mask` — callers must not read a value without
    checking the mask, since a zero there is "unobserved", not "zero bookings".
    """

    values: np.ndarray  # (n_arrival, n_dta), float
    observed_mask: np.ndarray  # (n_arrival, n_dta), bool
    arrival_dates: np.ndarray  # (n_arrival,), datetime64[D]
    dta_grid: np.ndarray  # (n_dta,), int
    as_of: np.datetime64
    target: np.ndarray | None  # (n_arrival,), float — e.g. unconstrained_demand; None at inference


def build_window(
    on_books: pd.DataFrame,
    as_of: np.datetime64 | pd.Timestamp,
    arrival_dates: Iterable[pd.Timestamp],
    dta_grid: Iterable[int],
    value_col: str = "otb",
    target: pd.Series | None = None,
) -> SurfaceWindow:
    """Slice `on_books` (one property's rows, or already pre-filtered) into a `SurfaceWindow`.

    `on_books` must have `stay_date`, `dta`, `booking_date`, and `value_col`. The mask is
    computed from `booking_date <= as_of` directly — never inferred from `dta` alone, per the
    module docstring's leakage rule — so it is correct even where `dta_grid` is the bucketed,
    non-consecutive grid.
    """
    as_of = pd.Timestamp(as_of)
    arrival_dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in arrival_dates))
    dta_grid = np.asarray(sorted(dta_grid), dtype=int)

    sub = on_books[
        on_books["stay_date"].isin(arrival_dates) & on_books["dta"].isin(dta_grid)
    ]
    pivoted = sub.pivot_table(index="stay_date", columns="dta", values=value_col, aggfunc="first")
    pivoted = pivoted.reindex(index=arrival_dates, columns=dta_grid)
    values = pivoted.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    # the leakage-safe mask: booking_date = arrival_date - dta, computed directly, not from dta
    arrival_col = np.repeat(arrival_dates.to_numpy()[:, None], len(dta_grid), axis=1)
    dta_col = np.tile(dta_grid[None, :], (len(arrival_dates), 1))
    booking_date = arrival_col - dta_col.astype("timedelta64[D]")
    observed_mask = booking_date <= np.datetime64(as_of, "D")

    values = np.where(observed_mask, values, 0.0)

    target_arr = None
    if target is not None:
        target_arr = target.reindex(arrival_dates).to_numpy(dtype=float)

    return SurfaceWindow(
        values=values, observed_mask=observed_mask, arrival_dates=arrival_dates.to_numpy(),
        dta_grid=dta_grid, as_of=np.datetime64(as_of, "D"), target=target_arr,
    )


def iter_training_windows(
    on_books: pd.DataFrame,
    arrival_dates: Iterable[pd.Timestamp],
    dta_grid: Iterable[int],
    as_of_dates: Iterable[pd.Timestamp],
    value_col: str = "otb",
    target: pd.Series | None = None,
) -> Iterator[SurfaceWindow]:
    """One `SurfaceWindow` per `as_of` in `as_of_dates`, same arrival/DTA grid each time."""
    arrival_dates = list(arrival_dates)
    dta_grid = list(dta_grid)
    for as_of in as_of_dates:
        yield build_window(on_books, as_of, arrival_dates, dta_grid, value_col, target)
