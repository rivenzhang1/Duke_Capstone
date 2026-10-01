"""The booking-pace function: how a stay date's demand is spread over days-to-arrival.

Spec: docs/simulation-spec.md.

The curve is produced **directly**, as an intensity over DTA, rather than emerging from a
simulated stream of reservations. Composition, per stay date:

    nu(k) = lam * shape(k) * shock(k)          E[sum_k nu(k)] = lam, exactly

`shape` carries everything systematic — the three-component lead-time mixture, the compression
shift, the booking-day-of-week ripple and promo diagonals. `shock` carries what makes this date
differ from the average date with the same lam.

## Why the shock is not renormalised

The obvious way to perturb a curve is to deform the pmf and renormalise, holding the total fixed.
That is wrong here, and wrong in an instructive way: with mass conserved, running hot at DTA 60
*forces* running cold later, so early pace is compensatory rather than predictive — the opposite
of the momentum real booking curves have.

So `shock` multiplies the intensity and the total moves with it. A date whose field runs high
over the middle of its window both books faster there and finishes higher. The
`exp(z - sigma^2/2)` correction keeps `E[shock] = 1`, so `lam_true` remains exactly expected
room-nights and stays usable as a latent to regress against.

## The two knobs, and what separates them

`shock_sigma` sets how far a date deviates. `shock_corr_len` sets over how many days of DTA that
deviation persists, which is what decides how much of it is **visible early**. That pair is the
whole forecastability dial: at a long correlation length a partial curve carries real information
about the final number; near zero the deviation decorrelates and the partial curve says nothing
beyond the level, however large `shock_sigma` is. The v1 generator sat at the second extreme
without a way to move — Poisson splitting makes increments independent across DTA by
construction — which capped what any pace-aware architecture could be shown to do.

Public surface:
    mixture_weights(d, occ_expected, is_event, respond) -> np.ndarray
    compression_shift(occ_expected, kappa) -> float
    booking_calendar_factor(stay_date, k_grid, promo_dates, book_dow) -> np.ndarray
    pace_shape(k_grid, weights, occ_expected, kappa, calendar_factor) -> np.ndarray
    pace_shock(rng, k_grid, sigma, corr_len) -> np.ndarray
    booking_intensity(lam, shape, shock) -> np.ndarray
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from deeppace_sim.config import BASE_MIX, COMPONENTS, GeneratorConstants

BOOK_DOW = np.array(GeneratorConstants.book_dow)  # Mon..Sun


def mixture_weights(
    d: date,
    occ_expected: float,
    is_event: bool,
    respond: bool = True,
    base_mix: dict[str, tuple[float, float, float]] = BASE_MIX,
) -> np.ndarray:
    """Segment mix for this stay date. Sums to 1.

    The base mix is portfolio-wide: archetype does **not** enter here. Everything that moves the
    mixture is regime — whether the arrival is a weekend, whether an event is on, and how full the
    date expects to be. Archetypes still end up with different curves, but through `occ_expected`
    rather than by stipulation. See the note on `config.BASE_MIX`.

    Events add a large early group block; high expected occupancy shifts weight toward group and
    leisure and away from last-minute. `respond=False` freezes the base mix.
    """
    w = np.array(base_mix["weekend" if d.weekday() >= 4 else "weekday"], dtype=float)
    if not respond:
        return w / w.sum()

    if is_event:
        w = w + np.array([0.35, 0.00, -0.10])

    c = np.clip((occ_expected - 0.70) / 0.25, -1, 1)
    w = w + np.array([0.10, 0.05, -0.15]) * c

    w = np.clip(w, 0.01, None)
    return w / w.sum()


def compression_shift(occ_expected: float, kappa: float = 0.35) -> float:
    """Multiplies every component's median lead time.

    Above 1 on dates expecting to be full, below 1 on soft ones — travellers who expect a sell-out
    commit earlier. Note what this does and does not do: it front-loads the demand signal, so peak
    and trough dates become *more* separable at long lead, not less. The v1 ablation measured this
    directly (check M2) against a design doc that claimed the opposite.
    """
    return 1.0 + kappa * np.clip((occ_expected - 0.70) / 0.25, -1, 1)


def lognormal_pmf(k: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Discretised log-normal density evaluated on the DTA grid."""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    mask = k > 0
    out[mask] = np.exp(-((np.log(k[mask]) - mu) ** 2) / (2 * sigma**2)) / (
        k[mask] * sigma * np.sqrt(2 * np.pi)
    )
    return out


def booking_calendar_factor(
    stay_date: date,
    k_grid: np.ndarray,
    promo_dates: dict[date, float],
    book_dow: np.ndarray | tuple[float, ...] = BOOK_DOW,
) -> np.ndarray:
    """The weekly booking-DOW ripple times any promo multiplier, indexed by DTA.

    Indexed on `booking_date = stay_date - k`, so its phase along the DTA axis depends on the stay
    date's own weekday. That is what makes both effects **diagonal** in `(arrival_date, DTA)`
    space rather than fixed per column, and invisible to features built per cell independently.
    """
    bdates = [stay_date - timedelta(days=int(k)) for k in k_grid]
    g = np.asarray(book_dow, dtype=float)[np.array([b.weekday() for b in bdates])]
    return g * np.array([promo_dates.get(b, 1.0) for b in bdates])


def pace_shape(
    k_grid: np.ndarray,
    weights: np.ndarray,
    occ_expected: float,
    kappa: float = 0.35,
    calendar_factor: np.ndarray | None = None,
) -> np.ndarray:
    """The systematic booking-pace pmf over DTA — everything except this date's own deviation.

    Normalised to sum to 1, so it is a pure *shape*: the level lives in `lam` and the deviation in
    `pace_shock`.
    """
    shift = compression_shift(occ_expected, kappa)
    pdf = np.zeros(len(k_grid), dtype=float)
    for w, cp in zip(weights, COMPONENTS.values(), strict=True):
        pdf += w * lognormal_pmf(k_grid + 0.5, cp.mu + np.log(shift), cp.sigma)
    if calendar_factor is not None:
        pdf = pdf * calendar_factor
    total = pdf.sum()
    return pdf / total if total > 0 else pdf


def pace_shock(
    rng: np.random.Generator,
    k_grid: np.ndarray,
    sigma: float,
    corr_len: float,
) -> np.ndarray:
    """This stay date's own deviation from the systematic shape. Mean 1 by construction.

    A log-space AR(1) walked from the far end of the booking window inwards, so `corr_len` is a
    persistence in days of DTA. `exp(z - sigma^2/2)` makes `E[shock] = 1` at every k, which is
    what keeps `sum_k lam * shape(k) * shock(k)` unbiased for `lam`.

    Innovations are drawn even when `sigma` is zero, so the ablation does not shift the property's
    random stream.
    """
    n = len(k_grid)
    innovations = rng.standard_normal(n)
    if sigma <= 0:
        return np.ones(n)

    rho = float(np.exp(-1.0 / max(corr_len, 1e-9)))
    z = np.empty(n)
    z[-1] = innovations[-1] * sigma  # k_max end: start at the stationary sd
    step = sigma * np.sqrt(max(1.0 - rho**2, 0.0))
    for i in range(n - 2, -1, -1):  # walk inwards, k_max -> 0
        z[i] = rho * z[i + 1] + step * innovations[i]
    return np.exp(z - sigma**2 / 2)


def booking_intensity(lam: float, shape: np.ndarray, shock: np.ndarray) -> np.ndarray:
    """Expected rooms booked at each DTA. Sums to `lam` in expectation, not exactly per draw."""
    return lam * shape * shock


def shock_bands(shock: np.ndarray, k_grid: np.ndarray, edges=(7, 30, 90)) -> tuple[float, ...]:
    """Mean shock over a few DTA bands — the persisted, low-dimensional view of the field.

    Storing the whole field would be one float per (stay date, DTA) cell. These four numbers are
    enough for the diagnostic that matters: whether a model's learned pace state correlates with
    the deviation actually applied.
    """
    bounds = [-1, *edges, int(k_grid.max())]
    return tuple(
        float(shock[(k_grid > lo) & (k_grid <= hi)].mean())
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    )
