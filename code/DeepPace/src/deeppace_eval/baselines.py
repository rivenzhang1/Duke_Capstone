"""1D baselines — the bar DeepPace-Net must clear.

Design: docs/design.md §6. All share one interface so scoring is identical across them, and all
consume the same `PaceData`/`PaceBatch` the four-case benchmark uses (`deeppace_net.pace_data`,
`deeppace_eval.covariates`) — so a baseline's numbers and a candidate's numbers can never
silently drift apart by being computed from different data plumbing.

    naive_pickup    additive pickup from the same weekday 52 weeks earlier;
                    the industry default, reported for context
    seasonal_reg    Ridge regression on calendar + current-OTB-fraction features;
                    tests whether a handful of hand-built features suffice
    flat_gbm        gradient boosting on the flattened, bucketed curve  <- the one that matters
    seq_1d          Holt's linear (level+trend) smoothing on daily rooms-sold history alone,
                    no DTA/OTB information at all; isolates the value of pace as such

`flat_gbm` receives exactly the same numbers as `two_stage`/`dynamic_factor` with the geometry
destroyed (OTB and net-change at a bucketed DTA grid, flattened into one feature vector per
row). Beating it *is* the hypothesis test; beating only `naive_pickup` demonstrates nothing
about the 2D claim.

lightgbm/sklearn are optional deps (the `eval` extra) — imported lazily inside each fitter, so
importing this module never requires them.

Public surface:
    BaselineResult (dataclass), BaselineConfig (dataclass), FittedBaseline
    naive_pickup(data, batch, weeks_back) -> BaselineResult
    seq_1d(data, batch, alpha, beta) -> BaselineResult
    fit_seasonal_reg(train_batches, ridge_alpha) -> FittedBaseline
    fit_flat_gbm(data, train_batches, grid, n_estimators, num_leaves, seed) -> FittedBaseline
    fit_predict(name, data, train_batches, test_batch, config) -> BaselineResult
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deeppace_eval.covariates import COMPACT_GRID
from deeppace_net.pace_data import PaceBatch, PaceData

NAMES = ("naive_pickup", "seasonal_reg", "flat_gbm", "seq_1d")


@dataclass(frozen=True)
class BaselineResult:
    name: str
    version: int
    prediction: np.ndarray  # raw, unclipped, aligned with batch.rows, in rooms


@dataclass(frozen=True)
class BaselineConfig:
    weeks_back: int = 52
    alpha: float = 0.3  # seq_1d level smoothing
    beta: float = 0.1  # seq_1d trend smoothing
    min_history_days: int = 14  # seq_1d: below this, fall back to current_otb
    ridge_alpha: float = 1.0
    gbm_estimators: int = 300
    gbm_leaves: int = 15
    gbm_grid: tuple[int, ...] = COMPACT_GRID
    seed: int = 20260826


class FittedBaseline:
    """Wraps a fitted model so `seasonal_reg`/`flat_gbm` share `naive_pickup`/`seq_1d`'s call
    shape (`.predict(batch) -> BaselineResult`) without needing a common base class."""

    def __init__(self, name: str, version: int, predict_fn: Callable[[PaceBatch], np.ndarray]):
        self.name = name
        self.version = version
        self._predict_fn = predict_fn

    def predict(self, batch: PaceBatch) -> BaselineResult:
        prediction = np.asarray(self._predict_fn(batch), dtype=np.float64)
        return BaselineResult(self.name, self.version, prediction)


def naive_pickup(data: PaceData, batch: PaceBatch, weeks_back: int = 52) -> BaselineResult:
    """Additive pickup from the same weekday `weeks_back` weeks earlier — the industry default.

    pickup = (that date's resolved final total) - (that date's OTB at the same DTA-to-arrival).
    prediction = current_otb + pickup. Weeks, not years, so the booking-DOW ripple and the
    arrival weekday stay aligned. Falls back to `current_otb` (zero pickup) wherever the prior
    date has no resolved history — a young property, or a date before the dataset starts.
    """
    rows = batch.rows
    prior_date = rows.stay_date - pd.Timedelta(weeks=weeks_back)
    otb_keys = pd.MultiIndex.from_tuples(
        list(zip(rows.property_id, prior_date, rows.horizon, strict=True))
    )
    prior_otb = data.surface.reindex(otb_keys).to_numpy(dtype=np.float64)
    final_keys = pd.MultiIndex.from_tuples(
        list(zip(rows.property_id, prior_date, strict=True))
    )
    prior_final = (
        data.history.set_index(["property_id", "stay_date"])["rooms_sold"]
        .reindex(final_keys)
        .to_numpy(dtype=np.float64)
    )
    pickup = prior_final - prior_otb
    pickup = np.where(np.isfinite(pickup), pickup, 0.0)
    prediction = rows.current_otb.to_numpy(dtype=np.float64) + pickup
    return BaselineResult("naive_pickup", 1, prediction)


def seq_1d(
    data: PaceData, batch: PaceBatch, alpha: float = 0.3, beta: float = 0.1,
    min_history_days: int = 14,
) -> BaselineResult:
    """Holt's linear (level+trend) smoothing on daily rooms-sold history alone.

    No DTA/OTB reaches this at all — by construction, it isolates whatever value the
    booking-pace surface adds over a classical history-only method. (Case A in the four-case
    benchmark asks a related question with a foundation model instead of a classical one.)
    Smoothing state is computed once per (property, as_of) and reused across horizons sharing
    that origin, since Holt's method doesn't depend on the horizon being forecast.
    """
    rows = batch.rows
    hist = data.history.set_index(["property_id", "stay_date"])["rooms_sold"].sort_index()
    known_properties = set(hist.index.get_level_values(0))
    cache: dict[tuple[int, pd.Timestamp], tuple[float, float] | None] = {}
    prediction = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows.itertuples()):
        key = (row.property_id, row.as_of)
        if key not in cache:
            series = hist.loc[row.property_id] if row.property_id in known_properties else pd.Series(dtype=float)
            series = series[series.index <= row.as_of].sort_index()
            if len(series) < min_history_days:
                cache[key] = None
            else:
                level, trend = float(series.iloc[0]), 0.0
                for v in series.iloc[1:]:
                    new_level = alpha * v + (1 - alpha) * (level + trend)
                    trend = beta * (new_level - level) + (1 - beta) * trend
                    level = new_level
                cache[key] = (level, trend)
        state = cache[key]
        prediction[i] = row.current_otb if state is None else state[0] + row.horizon * state[1]
    return BaselineResult("seq_1d", 1, prediction)


def _seasonal_features(batch: PaceBatch) -> np.ndarray:
    otb_fraction = (batch.rows.current_otb / batch.rows.capacity).to_numpy(dtype=np.float32)
    return np.concatenate([batch.context, otb_fraction[:, None]], axis=1)


def fit_seasonal_reg(train_batches: list[PaceBatch], ridge_alpha: float = 1.0) -> FittedBaseline:
    """Ridge regression on calendar + current-OTB-fraction — tests whether a handful of
    hand-built features suffice, without the DTA axis's full geometry."""
    from sklearn.linear_model import Ridge

    x = np.concatenate([_seasonal_features(b) for b in train_batches], axis=0)
    y = np.concatenate(
        [(b.rows.target / b.rows.capacity).to_numpy(dtype=np.float32) for b in train_batches]
    )
    model = Ridge(alpha=ridge_alpha)
    model.fit(x, y)

    def predict_fn(batch: PaceBatch) -> np.ndarray:
        fraction = model.predict(_seasonal_features(batch))
        return fraction * batch.rows.capacity.to_numpy(dtype=np.float64)

    return FittedBaseline("seasonal_reg", 1, predict_fn)


def _bucketed_positions(data: PaceData, grid: tuple[int, ...]) -> np.ndarray:
    """`batch.curves` rows are ordered by DTA descending (`k, k-1, ..., 0`); position of DTA
    value `d` is `k - d`."""
    if min(grid) < 1 or max(grid) > data.k:
        raise ValueError(f"grid must be within 1..{data.k}")
    return np.array([data.k - d for d in grid])


def _flat_gbm_features(batch: PaceBatch, positions: np.ndarray) -> np.ndarray:
    """The surface, flattened: OTB/capacity and net-change/capacity at the bucketed DTA grid,
    plus whether each of those cells was actually observed, plus the same calendar/horizon
    context the pace models see. This is deliberately *not* structured as (arrival_date, DTA) —
    the whole point is that the geometry is gone."""
    values = batch.curves[:, positions, 0]
    deltas = batch.curves[:, positions, 1]
    observed = batch.observed[:, positions].astype(np.float32)
    return np.concatenate([values, deltas, observed, batch.context], axis=1)


def fit_flat_gbm(
    data: PaceData,
    train_batches: list[PaceBatch],
    grid: tuple[int, ...] = COMPACT_GRID,
    n_estimators: int = 300,
    num_leaves: int = 15,
    seed: int = 20260826,
) -> FittedBaseline:
    """Gradient boosting on the flattened, bucketed curve — the strongest 1D contender.

    Receives exactly the numbers `two_stage`'s local GRU receives, with the geometry destroyed
    (no sequence, no (arrival_date, DTA) structure — just a flat feature vector). Beating this
    is design.md §6's actual hypothesis test.
    """
    from lightgbm import LGBMRegressor

    positions = _bucketed_positions(data, grid)
    x = np.concatenate([_flat_gbm_features(b, positions) for b in train_batches], axis=0)
    y = np.concatenate(
        [(b.rows.target / b.rows.capacity).to_numpy(dtype=np.float32) for b in train_batches]
    )
    model = LGBMRegressor(
        n_estimators=n_estimators, num_leaves=num_leaves, random_state=seed, verbosity=-1,
    )
    model.fit(x, y)

    def predict_fn(batch: PaceBatch) -> np.ndarray:
        fraction = model.predict(_flat_gbm_features(batch, positions))
        return fraction * batch.rows.capacity.to_numpy(dtype=np.float64)

    return FittedBaseline("flat_gbm", 1, predict_fn)


def fit_predict(
    name: str,
    data: PaceData,
    train_batches: list[PaceBatch],
    test_batch: PaceBatch,
    config: BaselineConfig = BaselineConfig(),
) -> BaselineResult:
    """Dispatch by name. `train_batches` is ignored by baselines that need no fitting
    (`naive_pickup`, `seq_1d`) — they're computed fresh from `data` at every call."""
    if name == "naive_pickup":
        return naive_pickup(data, test_batch, config.weeks_back)
    if name == "seq_1d":
        return seq_1d(data, test_batch, config.alpha, config.beta, config.min_history_days)
    if name == "seasonal_reg":
        if not train_batches:
            raise ValueError("seasonal_reg requires at least one training batch")
        return fit_seasonal_reg(train_batches, config.ridge_alpha).predict(test_batch)
    if name == "flat_gbm":
        if not train_batches:
            raise ValueError("flat_gbm requires at least one training batch")
        fitted = fit_flat_gbm(
            data, train_batches, config.gbm_grid, config.gbm_estimators,
            config.gbm_leaves, config.seed,
        )
        return fitted.predict(test_batch)
    raise KeyError(f"unknown baseline {name!r}; available: {NAMES}")
