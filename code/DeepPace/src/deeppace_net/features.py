"""Covariates aligned to the pace surface.

Design: docs/design.md §4.4.

Context is not optional here. At long DTA a peak date and a trough date can show the same
on-books while meaning opposite things; only context disambiguates them (§2.2). Minimum set:

    arrival-date axis   DOW, day-of-year / month, holiday flag, event flag
    booking-date axis   booking DOW (the 7-day ripple) — this is *diagonal* in
                        (arrival_date, dta) space and must be indexed as such
    property            archetype, capacity, base ADR

Note the booking-axis covariate is the one a per-cell feature builder silently drops: it is a
function of `booking_date = arrival_date - dta`, not of either axis alone.

`is_promo` is deliberately **not** included here even though a real operator often knows their
own promo calendar in advance. Leaving it out means the shared/global state a `dynamic_factor`
variant learns has to earn the diagonal promo signal from the observed increments themselves,
which is the harder and more informative test of whether the architecture reads the surface
(design.md §7.4) rather than a lookup.

ADR is available but **endogenous** (simulation-spec.md §7): it is computed from realized
on-books, so a model that uses it as an exogenous input learns "high rate -> high demand".
Include it only in variants that are explicitly testing endogeneity resistance — off by default.

Public surface:
    FeatureBundle (dataclass) — arrival_covs, booking_covs, property_covs
    build_features(window, property_row, event_calendar) -> FeatureBundle
    arrival_covariate_names / booking_covariate_names / property_covariate_names
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deeppace_sim.calendars import is_holiday
from deeppace_sim.config import ARCHETYPES
from deeppace_net.windows import SurfaceWindow

ARCHETYPE_NAMES = tuple(ARCHETYPES.keys())


def arrival_covariate_names() -> tuple[str, ...]:
    return ("dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_event")


def booking_covariate_names() -> tuple[str, ...]:
    return ("booking_dow_sin", "booking_dow_cos")


def property_covariate_names() -> tuple[str, ...]:
    return (*ARCHETYPE_NAMES, "capacity")


@dataclass(frozen=True)
class FeatureBundle:
    """Covariates for one `SurfaceWindow`. Diagonal covariates keep the (arrival, dta) shape;
    per-arrival-date and per-property covariates are broadcastable against it."""

    arrival_covs: np.ndarray  # (n_arrival, n_arrival_features)
    booking_covs: np.ndarray  # (n_arrival, n_dta, n_booking_features) — indexed by booking_date
    property_covs: np.ndarray  # (n_property_features,) — one property per window


def _cyclic(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * x / period
    return np.sin(angle), np.cos(angle)


def build_features(
    window: SurfaceWindow,
    property_row: pd.Series,
    event_calendar: dict | None = None,
) -> FeatureBundle:
    """`property_row` is one row of `properties.parquet` (archetype, capacity, ...).
    `event_calendar` maps `date -> event_mult` for this property (from `event_calendar.parquet`);
    only its keys are used here (an event flag, not the multiplier — the multiplier is what the
    model must ultimately explain, not be handed)."""
    arrival_dates = pd.DatetimeIndex(window.arrival_dates)
    event_calendar = event_calendar or {}

    dow_sin, dow_cos = _cyclic(arrival_dates.weekday.to_numpy(), 7)
    doy_sin, doy_cos = _cyclic(arrival_dates.dayofyear.to_numpy(), 365.25)
    holiday_flag = np.array([float(is_holiday(d.date())) for d in arrival_dates])
    event_flag = np.array([float(d.date() in event_calendar) for d in arrival_dates])
    arrival_covs = np.stack(
        [dow_sin, dow_cos, doy_sin, doy_cos, holiday_flag, event_flag], axis=1
    )

    booking_date = (
        arrival_dates.to_numpy()[:, None] - window.dta_grid[None, :].astype("timedelta64[D]")
    )
    booking_weekday = (
        (booking_date.astype("datetime64[D]").astype(np.int64) + 3) % 7
    )  # epoch (1970-01-01) was a Thursday; +3 realigns to Monday=0
    b_sin, b_cos = _cyclic(booking_weekday.astype(float), 7)
    booking_covs = np.stack([b_sin, b_cos], axis=-1)

    archetype_onehot = np.array(
        [float(property_row["archetype"] == a) for a in ARCHETYPE_NAMES]
    )
    property_covs = np.concatenate([archetype_onehot, [float(property_row["capacity"])]])

    return FeatureBundle(
        arrival_covs=arrival_covs, booking_covs=booking_covs, property_covs=property_covs,
    )
