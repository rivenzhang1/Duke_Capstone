"""Portfolio driver: compose the demand function and the pace function into a surface.

Spec: docs/simulation-spec.md.

Per property, per stay date:

    demand.demand_intensity          -> lam            (level: seasonality, events, mood, trend)
    pace.mixture_weights             -> w              (segment mix — regime only, not archetype)
    pace.pace_shape                  -> shape(k)       (systematic curve: mixture, compression,
                                                        booking-DOW ripple, promo diagonals)
    pace.pace_shock                  -> shock(k)       (this date's own deviation)
    pace.booking_intensity           -> nu(k)          = lam * shape * shock
    curve.draw_increments            -> gross(k)       (integer rooms, irreducible noise)
    curve.draw_cancellations         -> wash
    curve.apply_capacity             -> otb, gross_cum, cancel_cum, turned_away

`lam` reaches the curve's *shape* through one channel only — `occ_expected`, the expected *net*
fullness — which feeds both the segment mix and the compression shift. That coupling is deliberate: without
it the fraction booked at a given DTA would say nothing about how full the date ends up, and the
classical pickup baselines, which assume curve shape is common across dates, would be
near-optimal by construction.

RNG layout: two portfolio-wide streams at **fixed indices 0 and 1** (property ids, shared promo
calendar), then one independent stream per property. The indices are fixed so that changing
`n_props` cannot shift them, and ids come from a truncated permutation rather than
`choice(size=n)`, whose first n outputs depend on n. Both were bugs in v1: growing the portfolio
silently regenerated the promo calendar and renamed every property, which made "adding a property
does not perturb the others" true of the draws and false of the dataset.

Public surface:
    PropertyState, StayDateResult (dataclasses)
    portfolio_streams(config) -> (id_rng, promo_rng, per-property rngs)
    promo_calendar(config) -> dict[date, float]
    simulate_portfolio(config) -> iterator of (PropertyState, list[StayDateResult])
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from deeppace_sim.calendars import build_event_calendar, build_promo_calendar, is_holiday
from deeppace_sim.config import ARCHETYPES, SimConfig
from deeppace_sim.curve import (
    adr_at_dta,
    apply_capacity,
    draw_cancellations,
    draw_increments,
)
from deeppace_sim.demand import demand_intensity, mood_path
from deeppace_sim.pace import (
    booking_calendar_factor,
    booking_intensity,
    mixture_weights,
    pace_shape,
    pace_shock,
    shock_bands,
)

_ID_STREAM = 0
_PROMO_STREAM = 1
_N_SHARED_STREAMS = 2


@dataclass(frozen=True)
class PropertyState:
    property_id: int
    archetype: str
    capacity: int
    base_adr: float
    base_demand: float
    event_calendar: dict[date, float]
    mood: np.ndarray


@dataclass(frozen=True)
class StayDateResult:
    property_id: int
    stay_date: date
    dow: int
    otb: np.ndarray
    gross_cum: np.ndarray
    cancel_cum: np.ndarray
    adr_by_k: np.ndarray
    rooms_sold: int
    unconstrained_demand: int
    turned_away: int
    sold_out: bool
    was_censored: bool
    lam_true: float
    is_event: bool
    is_holiday: bool
    weights: np.ndarray  # w_group, w_leisure, w_lastmin
    event_mult: float
    mood: float
    gross_drawn: int
    shock_total: float
    """Realised total demand as a multiple of `lam_true` — the date-level pace shock, summarised."""
    shock_bands: tuple[float, ...]
    """Mean shock over DTA bands (0-7, 8-30, 31-90, 91+). The recoverable form of the field."""


def portfolio_streams(config: SimConfig) -> tuple[np.random.Generator, np.random.Generator, list]:
    """(id_rng, promo_rng, per-property rngs) — the one place the stream layout is defined."""
    #children = np.random.default_rng(config.seed).spawn(config.n_props + _N_SHARED_STREAMS)
    # SeedSequence.spawn rather than Generator.spawn (numpy >= 1.25 only): identical streams
    seeds = np.random.SeedSequence(config.seed).spawn(config.n_props + _N_SHARED_STREAMS)                                                                                                           
    children = [np.random.default_rng(s) for s in seeds]
    return children[_ID_STREAM], children[_PROMO_STREAM], list(children[_N_SHARED_STREAMS:])


def promo_calendar(config: SimConfig) -> dict[date, float]:
    """The planted flash-sale booking dates — ground truth for the diagonal checks.

    Built through this one function by both the generator and the writer, so the persisted
    calendar cannot drift from the one the bookings were drawn against.
    """
    c = config.constants
    _, promo_rng, _ = portfolio_streams(config)
    return build_promo_calendar(
        promo_rng,
        config.start - timedelta(days=config.k_max),
        config.end,
        c.n_promos,
        c.promo_mult,
        c.promo_max_days,
    )


def simulate_portfolio(config: SimConfig) -> Iterator[tuple[PropertyState, list[StayDateResult]]]:
    c = config.constants
    id_rng, _, prop_rngs = portfolio_streams(config)
    promo = promo_calendar(config)

    property_ids = id_rng.permutation(np.arange(10000, 100000))[: config.n_props]
    archetypes = list(ARCHETYPES.keys())
    n_days = (config.end - config.start).days + 1
    k_grid = np.arange(config.k_max + 1)

    for i in range(config.n_props):
        archetype = archetypes[i % len(archetypes)]
        rng = prop_rngs[i]
        A = ARCHETYPES[archetype]
        capacity = int(rng.integers(A.cap[0], A.cap[1] + 1))
        prop = PropertyState(
            property_id=int(property_ids[i]),
            archetype=archetype,
            capacity=capacity,
            base_adr=float(rng.uniform(*A.adr)),
            base_demand=A.base_demand,
            event_calendar=build_event_calendar(rng, A, config.start, config.end),
            mood=mood_path(rng, n_days, c.mood_phi, c.mood_sigma, c.mood_burn_in),
        )

        results = []
        for offset in range(n_days):
            d = config.start + timedelta(days=offset)
            event_mult = prop.event_calendar.get(d, 1.0)
            is_event = d in prop.event_calendar

            lam = demand_intensity(
                A, capacity, d, config.start, prop.mood, event_mult, rng, c.noise_sigma
            )
            # expected *net* fullness — see GeneratorConstants.expected_wash
            occ_expected = float(
                np.clip(lam * (1.0 - c.expected_wash) / capacity, 0.0, 1.3)
            )

            w = mixture_weights(d, occ_expected, is_event, respond=c.mixture_response)
            calendar = booking_calendar_factor(d, k_grid, promo, c.book_dow)
            shape = pace_shape(k_grid, w, occ_expected, c.kappa, calendar)
            shock = pace_shock(rng, k_grid, c.shock_sigma, c.shock_corr_len)
            nu = booking_intensity(lam, shape, shock)

            gross = draw_increments(rng, nu, c.increment_dispersion)
            cancel_origin, cancel_target = draw_cancellations(rng, gross, float(w[0]), c)
            curve = apply_capacity(
                gross, cancel_origin, cancel_target, capacity, rng, c.capacity_enforced
            )

            results.append(StayDateResult(
                property_id=prop.property_id, stay_date=d, dow=d.weekday(),
                otb=curve.otb, gross_cum=curve.gross_cum, cancel_cum=curve.cancel_cum,
                adr_by_k=adr_at_dta(prop.base_adr, capacity, curve.otb, c),
                rooms_sold=curve.rooms_sold,
                unconstrained_demand=curve.unconstrained_demand,
                turned_away=curve.turned_away,
                sold_out=curve.rooms_sold >= capacity,
                was_censored=curve.was_censored,
                lam_true=lam, is_event=is_event, is_holiday=is_holiday(d), weights=w,
                event_mult=event_mult, mood=float(prop.mood[offset]),
                gross_drawn=curve.gross_drawn,
                shock_total=float(curve.gross_drawn / lam) if lam > 0 else 1.0,
                shock_bands=shock_bands(shock, k_grid),
            ))

        yield prop, results
