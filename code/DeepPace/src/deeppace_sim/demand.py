"""Stay-date demand intensity lambda(p, d) — multiplicative in log space.

Spec: docs/simulation-spec.md.

Carried over from the v1 generator unchanged in form. It is the half of that design the
validation tier confirmed exactly: dividing every configured factor back out of `lam_true` left a
residual with mean -0.003 and sd 0.1136 against a configured 0.11, with no structure by month,
day-of-week or year. Only the *arrival process* below it was replaced.

    lambda = capacity * base_demand * yearly * monthly * dow * day_of_month
             * holiday * event * trend * noise

Three things here are load-bearing and should not be simplified away:

- **Yearly and monthly are separate on purpose.** Yearly is the smooth 2-harmonic annual
  cycle; monthly is the *non-smooth* month-of-year step structure. A Fourier series alone
  cannot produce a December 25% below November.
- **`mood`** is a slow AR(1) (phi=0.93, weekly innovations) shared across all stay dates of a
  property, creating persistent over/under-performance. Without it demand is independent
  day-to-day and every model looks artificially good.
- **The trend spans a genuine level shift** — post-COVID recovery ramp through 2023 settling
  into ~3%/yr growth. Train-2023/24, test-2025 must cope with it; do not smooth it away.

Every scalar here (`noise_sigma`, `mood_phi`, `mood_sigma`) comes from
`config.GeneratorConstants` so it is captured in the config snapshot and can be ablated.

Public surface:
    demand_factors(archetype_cfg, capacity, d, start, event_mult) -> dict[str, float]
    demand_intensity(archetype_cfg, capacity, d, start, mood, event_mult, rng, noise_sigma) -> float
    mood_path(rng, n_days, phi, sigma, burn_in) -> np.ndarray
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date

import numpy as np

from deeppace_sim.calendars import holiday_multiplier
from deeppace_sim.config import ArchetypeConfig


def mood_path(
    rng: np.random.Generator,
    n_days: int,
    phi: float = 0.93,
    sigma: float = 0.09,
    burn_in: bool = True,
) -> np.ndarray:
    """AR(1) weekly-innovation mood path, one value per day, shared across a property's stay dates.

    Innovations are drawn standard-normal and scaled, so ``sigma=0.0`` (the M7 ablation) removes
    the mood without changing how much randomness the property's stream consumes.

    ``burn_in`` starts the path at the stationary sd ``sigma/sqrt(1 - phi^2)`` rather than at a
    single innovation. v1 defaulted it off, which made its first ~40 weeks systematically less
    variable than the rest; it defaults on here.
    """
    n_weeks = n_days // 7 + 2
    innovations = rng.normal(0.0, 1.0, size=n_weeks) * sigma
    weekly = np.zeros(n_weeks)
    weekly[0] = innovations[0] / np.sqrt(1.0 - phi**2) if burn_in and phi < 1.0 else innovations[0]
    for i in range(1, n_weeks):
        weekly[i] = phi * weekly[i - 1] + innovations[i]
    return np.repeat(weekly, 7)[:n_days]


def demand_factors(
    archetype_cfg: ArchetypeConfig,
    capacity: float,
    d: date,
    start: date,
    event_mult: float = 1.0,
) -> dict[str, float]:
    """Every *deterministic* factor of lambda, separately.

    One source for the factor stack, used three ways: `demand_intensity` multiplies them, the
    decomposition figure plots them, and a parameter-recovery check divides them back out of
    `lam_true` to confirm nothing is left but the noise draw. In v1 the recovery module
    reimplemented this stack, so the check and the generator could silently disagree about what
    the model was.
    """
    A = archetype_cfg
    doy = d.timetuple().tm_yday
    dim = _calendar.monthrange(d.year, d.month)[1]
    t = (d - start).days

    return {
        "level": capacity * A.base_demand,
        "yearly": float(np.exp(sum(
            A.yearly["a"][h] * np.cos(2 * np.pi * (h + 1) * doy / 365.25)
            + A.yearly["b"][h] * np.sin(2 * np.pi * (h + 1) * doy / 365.25)
            for h in range(2)
        ))),
        "monthly": float(A.month[d.month - 1]),
        "dow": float(A.dow[d.weekday()]),
        "day_of_month": 1.0 + A.month_end_push * (
            1.0 if (d.day >= dim - 2 or d.day <= 2) else -0.25
        ),
        "holiday": float(holiday_multiplier(d, A.hol)),
        "event": float(event_mult),
        "trend": float((0.90 + 0.10 * (1 - np.exp(-t / 300.0))) * (1.03 ** (t / 365.25))),
    }


def demand_intensity(
    archetype_cfg: ArchetypeConfig,
    capacity: float,
    d: date,
    start: date,
    mood: np.ndarray,
    event_mult: float,
    rng: np.random.Generator,
    noise_sigma: float = 0.11,
) -> float:
    known = 1.0
    for value in demand_factors(archetype_cfg, capacity, d, start, event_mult).values():
        known *= value
    t = (d - start).days
    eps = float(np.exp(rng.normal(0.0, 1.0) * noise_sigma + mood[t]))
    return known * eps
