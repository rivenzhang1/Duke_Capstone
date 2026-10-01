"""The booking-pace surface reindexed to booking-date order, for the `dynamic_factor` variant.

Design: docs/design.md §12 (2026-09-18 entry, the dynamic-factor architecture discussion).

The other candidate encoders (`conv2d`, `axial`, `hier_state`) consume a `SurfaceWindow` — an
(arrival_date, DTA) patch sliced at one as-of date (`windows.py`). `dynamic_factor` needs a
different shape entirely: a *stream* of booking days, walked in calendar order, where each day
touches many arrival dates at once. That's the natural axis for it because the two states it
maintains are each keyed to a different one of the surface's structural facts (design.md §2):

    shared state    <- booking_calendar_factor(booking_date): book_dow ripple + promo, the
                       same multiplier applied to every arrival date booked that day
    local state     <- pace_shock(stay_date, k): an AR(1) field walked in booking-date order
                       for that one stay date — this is *already* a state-space process in the
                       generator, not an analogy

Every (property_id, stay_date) pair gets one **slot** — a fixed index into a context table
(archetype, capacity, arrival DOW/month/holiday/event) and a target (`unconstrained_demand`,
not `rooms_sold` — capacity/cancellation are out of scope for this variant; a plain
`min(prediction, capacity)` cap is applied, if at all, strictly as a separate post-processing
step, never inside the model). `is_promo` is deliberately absent from both slot context and day
context, for the same reason `features.py` leaves it out: the shared state has to earn the
diagonal signal from the observed increments, not be handed it.

Public surface:
    BookingDayBatch (dataclass) — one calendar day, vectorized across every slot it touches
    BookingStream (dataclass) — slots (context + target) + days (ascending booking_date)
    build_booking_stream(ds, start, end, property_ids, as_of) -> BookingStream
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from deeppace_sim.calendars import is_holiday
from deeppace_sim.config import ARCHETYPES
from deeppace_sim.dataset import LoadedDataset

ARCHETYPE_NAMES = tuple(ARCHETYPES.keys())
N_SLOT_FEATURES = len(ARCHETYPE_NAMES) + 1 + 6  # archetype one-hot + capacity + 6 arrival covs


def _cyclic(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * x / period
    return np.sin(angle), np.cos(angle)


@dataclass(frozen=True)
class BookingDayBatch:
    """Everything that happened, portfolio-wide, on one booking date."""

    booking_date: pd.Timestamp
    booking_dow_sin: float
    booking_dow_cos: float
    slot_idx: np.ndarray  # (n_active,) int — which slot each row belongs to
    dta: np.ndarray  # (n_active,) int
    gross_inc: np.ndarray  # (n_active,) float
    cancel_inc: np.ndarray  # (n_active,) float


@dataclass(frozen=True)
class BookingStream:
    """`slots` has one row per (property_id, stay_date): slot_idx, context columns, `target`.
    `days` is every booking day in `[start, end]` that has at least one active slot, ascending.
    """

    slots: pd.DataFrame
    days: list[BookingDayBatch] = field(default_factory=list)

    @property
    def slot_context(self) -> np.ndarray:
        """(n_slots, N_SLOT_FEATURES) — the fixed per-slot context, in a stable column order."""
        cols = [f"archetype_{a}" for a in ARCHETYPE_NAMES] + [
            "capacity", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_event",
        ]
        return self.slots[cols].to_numpy(dtype=float).copy()

    @property
    def target(self) -> np.ndarray:
        return self.slots["target"].to_numpy(dtype=float).copy()

    @property
    def capacity(self) -> np.ndarray:
        return self.slots["capacity"].to_numpy(dtype=float).copy()


def build_booking_stream(
    ds: LoadedDataset,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    property_ids: list[int] | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> BookingStream:
    """Reindex `ds` to booking-date order over `[start, end]` (arrival dates), for `property_ids`.

    `as_of`, if given, drops every row with `booking_date > as_of` — the same leakage rule
    `windows.py` enforces, applied to this shape instead. Leave it `None` to build the full
    stream (e.g. for a training split that is itself bounded by `end`, so every booking date in
    range is legitimately observable by the time its own arrival dates resolve).
    """
    inc = ds.increments()
    if property_ids is not None:
        inc = inc[inc["property_id"].isin(property_ids)]
    if start is not None:
        inc = inc[inc["stay_date"] >= pd.Timestamp(start)]
    if end is not None:
        inc = inc[inc["stay_date"] <= pd.Timestamp(end)]
    if as_of is not None:
        inc = inc[inc["booking_date"] <= pd.Timestamp(as_of)]

    slot_keys = (
        inc[["property_id", "stay_date"]].drop_duplicates().sort_values(["property_id", "stay_date"])
        .reset_index(drop=True)
    )
    slot_keys["slot_idx"] = np.arange(len(slot_keys))
    inc = inc.merge(slot_keys, on=["property_id", "stay_date"], how="left")

    props = ds.properties.set_index("property_id")
    archetype = slot_keys["property_id"].map(props["archetype"])
    for a in ARCHETYPE_NAMES:
        slot_keys[f"archetype_{a}"] = (archetype == a).astype(float)
    slot_keys["capacity"] = slot_keys["property_id"].map(props["capacity"]).astype(float)

    dow_sin, dow_cos = _cyclic(slot_keys["stay_date"].dt.weekday.to_numpy(), 7)
    doy_sin, doy_cos = _cyclic(slot_keys["stay_date"].dt.dayofyear.to_numpy(), 365.25)
    slot_keys["dow_sin"], slot_keys["dow_cos"] = dow_sin, dow_cos
    slot_keys["doy_sin"], slot_keys["doy_cos"] = doy_sin, doy_cos
    slot_keys["is_holiday"] = [float(is_holiday(d.date())) for d in slot_keys["stay_date"]]

    events = set(zip(ds.event_calendar["property_id"], ds.event_calendar["stay_date"], strict=True)) \
        if ds.event_calendar is not None else set()
    slot_keys["is_event"] = [
        float((p, d) in events) for p, d in zip(slot_keys["property_id"], slot_keys["stay_date"], strict=True)
    ]

    gt = ds.ground_truth.set_index(["property_id", "stay_date"])["unconstrained_demand"]
    slot_keys["target"] = (
        gt.reindex(list(zip(slot_keys["property_id"], slot_keys["stay_date"], strict=True))).to_numpy()
    )

    days: list[BookingDayBatch] = []
    for booking_date, day_df in inc.groupby("booking_date", sort=True):
        bd_sin, bd_cos = _cyclic(np.array([booking_date.weekday()], dtype=float), 7)
        days.append(BookingDayBatch(
            booking_date=booking_date,
            booking_dow_sin=float(bd_sin[0]), booking_dow_cos=float(bd_cos[0]),
            slot_idx=day_df["slot_idx"].to_numpy().copy(),
            dta=day_df["dta"].to_numpy(dtype=np.int64).copy(),
            gross_inc=day_df["gross_inc"].to_numpy(dtype=np.float64).copy(),
            cancel_inc=day_df["cancel_inc"].to_numpy(dtype=np.float64).copy(),
        ))

    return BookingStream(slots=slot_keys, days=days)
