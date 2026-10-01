"""Calendar effects: holidays, per-property event calendars, and promo diagonals.

Spec: docs/simulation-spec.md §4 (holidays, events), §5.3 (promo calendar).

Holiday effects **invert by archetype** — Christmas is ~0.28x for urban business and ~1.50x
for a resort. Any model that learns one global holiday effect is badly wrong on half the
portfolio, which is deliberate.

Promos are the diagonal test case: 18 flash-sale *booking* **events** over the 4-year booking
window (34 calendar days at the shipped defaults), multiplier 1.6-2.6, lasting 1-3 days each,
applied portfolio-wide. Because ``booking_date = arrival_date - dta`` they appear as diagonal
streaks in the surface and are invisible to features built per (arrival_date, dta) cell
(design.md §2.3).

Both calendars are persisted by ``writer.py``. A latent that is only *reconstructible* from the
seed is not ground truth: rebuilding one by replaying the spawn order couples the check to
generator internals and breaks silently when either changes.

Carried over unchanged from the v1 generator, where the recovery checks verified every factor of
lambda exactly (residual mean -0.003, sd 0.1136 against a configured 0.11).

Public surface:
    holiday_multiplier(d, archetype_holidays) -> float
    holiday_dates(year) -> dict[str, date]
    is_holiday(d) -> bool
    build_event_calendar(rng, archetype_cfg, start, end) -> dict[date, float]
    build_promo_calendar(rng, booking_start, booking_end, n_promos, mult_range, max_days)
        -> dict[date, float]
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

from deeppace_sim.config import ArchetypeConfig


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: 0=Mon..6=Sun. n: 1-indexed occurrence within the month."""
    weeks = _calendar.monthcalendar(year, month)
    days = [week[weekday] for week in weeks if week[weekday] != 0]
    return date(year, month, days[n - 1])


def _last_weekday(year: int, month: int, weekday: int) -> date:
    weeks = _calendar.monthcalendar(year, month)
    days = [week[weekday] for week in weeks if week[weekday] != 0]
    return date(year, month, days[-1])


@lru_cache(maxsize=None)
def holiday_dates(year: int) -> dict[str, date]:
    return {
        "mlk": _nth_weekday(year, 1, 0, 3),
        "memorial": _last_weekday(year, 5, 0),
        "july4": date(year, 7, 4),
        "labor": _nth_weekday(year, 9, 0, 1),
        "thanksgiving": _nth_weekday(year, 11, 3, 4),
        "christmas": date(year, 12, 25),
        "nye": date(year, 12, 31),
    }


# how many days on either side of the holiday the effect fades out over
_HOLIDAY_SPAN: dict[str, int] = {
    "mlk": 2, "memorial": 2, "july4": 1, "labor": 2,
    "thanksgiving": 2, "christmas": 3, "nye": 2,
}


def holiday_multiplier(d: date, archetype_holidays: dict[str, float]) -> float:
    """Multiplicative demand effect of nearby holidays, linearly fading to 1.0 at the span edge."""
    mult = 1.0
    for name, hol_mult in archetype_holidays.items():
        hd = holiday_dates(d.year).get(name)
        if hd is None:
            continue
        span = _HOLIDAY_SPAN.get(name, 1)
        dist = abs((d - hd).days)
        if dist <= span:
            blend = 1.0 - dist / (span + 1)
            mult *= 1.0 + (hol_mult - 1.0) * blend
    return mult


def is_holiday(d: date) -> bool:
    return d in holiday_dates(d.year).values()


def build_event_calendar(
    rng: np.random.Generator, archetype_cfg: ArchetypeConfig, start: date, end: date
) -> dict[date, float]:
    """Per-property known-in-advance event calendar — the group-booking driver (spec §5.1)."""
    n_events = int(rng.integers(archetype_cfg.events[0], archetype_cfg.events[1] + 1))
    n_days = (end - start).days + 1
    calendar_mult: dict[date, float] = {}
    for _ in range(n_events):
        offset = int(rng.integers(0, n_days))
        ev_date = start + timedelta(days=offset)
        mult = float(rng.uniform(*archetype_cfg.event_mult))
        duration = int(rng.integers(1, 4))
        for i in range(duration):
            d = ev_date + timedelta(days=i)
            calendar_mult[d] = max(calendar_mult.get(d, 1.0), mult)
    return calendar_mult


def build_promo_calendar(
    rng: np.random.Generator,
    booking_start: date,
    booking_end: date,
    n_promos: int = 18,
    mult_range: tuple[float, float] = (1.6, 2.6),
    max_days: int = 3,
) -> dict[date, float]:
    """Portfolio-wide flash-sale *booking* dates — the diagonal-streak test case (spec §5.3).

    ``n_promos`` counts promo *events*; each spans 1..``max_days`` days, so the returned mapping
    holds more keys than that (18 events -> 34 calendar days at the shipped defaults).
    ``n_promos=0`` is the M5 ablation.
    """
    n_days = (booking_end - booking_start).days + 1
    promo: dict[date, float] = {}
    for _ in range(n_promos):
        offset = int(rng.integers(0, n_days))
        p_date = booking_start + timedelta(days=offset)
        mult = float(rng.uniform(*mult_range))
        duration = int(rng.integers(1, max_days + 1))
        for i in range(duration):
            d = p_date + timedelta(days=i)
            promo[d] = max(promo.get(d, 1.0), mult)
    return promo
