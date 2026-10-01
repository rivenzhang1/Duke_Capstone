"""From booking intensity to the observed on-books curve: increments, cancellations, capacity.

Spec: docs/simulation-spec.md.

This is where the top-down generator stops being a formula and starts being a process. Three
steps, in order, and the order is the point:

1. **Increments.** Draw integer rooms booked at each DTA around the intensity, with `var/mean =
   increment_dispersion`. This is the irreducible noise — the part no model can recover — and it
   stands in for batch arrivals without simulating batches.
2. **Cancellations.** Each booking increment spawns cancellations at a rate that rises with its
   own lead time, landing later (closer to arrival) than the booking. Long-lead bookings wash
   hardest; group-heavy dates wash harder still.
3. **Capacity.** Walk the DTA axis from the far end inwards, accepting what fits. **Not** a clip
   on the final total.

## Why capacity is still walked

The v1 generator earned three things from enforcing capacity in time order, and none of them
survives clipping a finished curve: sell-out dates plateau early rather than climbing to the wall,
the observed final is genuinely right-censored, and `turned_away` is real rather than inferred.
Dropping reservation-level simulation does not require dropping the walk — it only changes what
the walk consumes, from individual bookings to per-DTA increments.

It also preserves the subtlety that a rejected booking is **never retried**. When capacity binds
at DTA k, the rooms turned away are gone; a cancellation at DTA k-20 frees space that nothing
reclaims. So a date can turn demand away and still finish below capacity — on the v1 dataset that
was 78% of all censored dates. `sold_out` is therefore not the censoring indicator, and
`was_censored = turned_away > 0` is what a censored likelihood must key on.

Public surface:
    draw_increments(rng, intensity, dispersion) -> np.ndarray
    draw_cancellations(rng, gross, w_group, constants) -> tuple[np.ndarray, np.ndarray]
    apply_capacity(gross, cancel_origin, cancel_target, capacity, rng, enforced) -> CurveResult
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deeppace_sim.config import GeneratorConstants


@dataclass(frozen=True)
class CurveResult:
    """One stay date's realised curve. Arrays are indexed by DTA, 0..k_max."""

    otb: np.ndarray  # net rooms on the books
    gross_cum: np.ndarray  # cumulative accepted bookings
    cancel_cum: np.ndarray  # cumulative realised cancellations
    rooms_sold: int
    unconstrained_demand: int  # net of cancellation, ignoring capacity
    turned_away: int
    gross_drawn: int  # rooms demanded before capacity, before cancellation

    @property
    def was_censored(self) -> bool:
        """Capacity actually bound. Not the same as `sold_out` — see the module docstring."""
        return self.turned_away > 0


def draw_increments(
    rng: np.random.Generator, intensity: np.ndarray, dispersion: float
) -> np.ndarray:
    """Integer rooms booked at each DTA, with `var/mean = dispersion` around `intensity`.

    `dispersion = 1` is Poisson counting. Above that, a negative binomial: for mean `m` and
    variance ratio `phi`, `p = 1/phi` and `n = m/(phi-1)` give exactly that mean and ratio. The
    excess over Poisson is what a lumpy arrival — a group block landing on one day — would have
    produced, without the machinery of drawing party sizes.
    """
    mean = np.clip(intensity, 0.0, None)
    if dispersion <= 1.0:
        return rng.poisson(mean).astype(np.int64)

    p = 1.0 / dispersion
    n = mean / (dispersion - 1.0)
    out = np.zeros(len(mean), dtype=np.int64)
    live = n > 0
    if live.any():
        out[live] = rng.negative_binomial(n[live], p)
    return out


def draw_cancellations(
    rng: np.random.Generator,
    gross: np.ndarray,
    w_group: float,
    constants: GeneratorConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Which bookings cancel, and when.

    Returns two parallel arrays: the DTA each cancelled room was *booked* at, and the DTA it
    cancels at. Keeping the origin is what lets `apply_capacity` drop the cancellations belonging
    to bookings it rejected — a booking that never happened cannot wash.

    Wash probability rises with the booking's own lead time and with how group-heavy the date is;
    timing is a Beta fraction of the lead time, skewed towards cancelling late.
    """
    c = constants
    k_grid = np.arange(len(gross))
    p_can = c.cancel_base + c.cancel_amp * (1.0 - np.exp(-k_grid / c.cancel_tau))
    p_can = np.minimum(p_can * (1.0 + c.cancel_group_extra * w_group), c.cancel_cap)

    n_can = rng.binomial(gross, p_can)
    total = int(n_can.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    origin = np.repeat(k_grid, n_can)
    frac = rng.beta(*c.cancel_timing_beta, size=total)
    target = np.floor(origin * frac).astype(np.int64)
    return origin, target


def apply_capacity(
    gross: np.ndarray,
    cancel_origin: np.ndarray,
    cancel_target: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
    enforced: bool = True,
) -> CurveResult:
    """Walk DTA from the far end inwards, accepting bookings while they fit.

    The unconstrained path is computed first and the walk is skipped when it never reaches the
    wall, which is the common case and keeps generation fast. When it does bind, rooms are turned
    away and the cancellations belonging to the rejected share are dropped with them.
    """
    k_max = len(gross) - 1
    cancel_all = np.bincount(cancel_target, minlength=k_max + 1)[: k_max + 1]
    net = gross - cancel_all
    # on-books at DTA k is everything that happened at DTA >= k
    unconstrained_path = np.cumsum(net[::-1])[::-1]
    unconstrained_final = int(gross.sum() - len(cancel_target))

    if not enforced or unconstrained_path.max(initial=0) <= capacity:
        return CurveResult(
            otb=unconstrained_path,
            gross_cum=np.cumsum(gross[::-1])[::-1],
            cancel_cum=np.cumsum(cancel_all[::-1])[::-1],
            rooms_sold=int(unconstrained_path[0]),
            unconstrained_demand=unconstrained_final,
            turned_away=0,
            gross_drawn=int(gross.sum()),
        )

    by_origin = [np.zeros(0, dtype=np.int64)] * (k_max + 1)
    if len(cancel_origin):
        order = np.argsort(cancel_origin, kind="stable")
        origins_sorted = cancel_origin[order]
        targets_sorted = cancel_target[order]
        bounds = np.searchsorted(origins_sorted, np.arange(k_max + 2))
        for k in range(k_max + 1):
            by_origin[k] = targets_sorted[bounds[k]:bounds[k + 1]]

    accepted = np.zeros(k_max + 1, dtype=np.int64)
    cancelled = np.zeros(k_max + 1, dtype=np.int64)
    pending = np.zeros(k_max + 1, dtype=np.int64)
    on_books = 0
    turned_away = 0

    for k in range(k_max, -1, -1):
        want = int(gross[k])
        if want:
            take = min(want, capacity - on_books)
            accepted[k] = take
            turned_away += want - take
            on_books += take
            targets = by_origin[k]
            if len(targets):
                # a rejected booking never happened, so it cannot wash: keep its cancellations
                # with the same probability that its booking was accepted
                keep = targets if take == want else targets[rng.random(len(targets)) < take / want]
                if len(keep):
                    np.add.at(pending, keep, 1)
        if pending[k]:
            cancelled[k] = pending[k]
            on_books -= int(pending[k])

    return CurveResult(
        otb=np.cumsum((accepted - cancelled)[::-1])[::-1],
        gross_cum=np.cumsum(accepted[::-1])[::-1],
        cancel_cum=np.cumsum(cancelled[::-1])[::-1],
        rooms_sold=int((accepted - cancelled).sum()),
        unconstrained_demand=unconstrained_final,
        turned_away=int(turned_away),
        gross_drawn=int(gross.sum()),
    )


def adr_at_dta(
    base_adr: float,
    capacity: int,
    otb: np.ndarray,
    constants: GeneratorConstants,
) -> np.ndarray:
    """Rate from realised on-books: a BAR ladder plus a close-in premium.

    Causally *downstream* of demand, deliberately. A naive regression learns "high rate implies
    high demand"; the dataset is built to punish exactly that. Elasticity is zero, so this tests
    endogeneity resistance, not pricing.
    """
    c = constants
    k = np.arange(len(otb))
    ladder = c.adr_ladder_base + c.adr_ladder_amp * (otb / capacity) ** c.adr_ladder_exp
    close_in = 1.0 + c.adr_closein_amp * np.exp(-k / c.adr_closein_tau)
    return base_adr * ladder * close_in
