"""Simulator configuration: archetypes, pace components, and every generator constant.

Spec: docs/simulation-spec.md.

A change here **bumps the dataset version**, and `generator_hash` digests every constant so a
result can name the generator that produced it rather than a version label someone typed.

## The knobs that matter, and why they are these three

The v1 generator drew reservations from a Poisson process. That made two properties of the
dataset consequences of the party-size distribution rather than things anyone chose:
overdispersion was pinned near `E[S²]/E[S]` by the group party-size range, and booking increments
at different DTA were independent by construction, so there was no pace momentum to learn. See
`deeppace_sim_v1/README.md`.

Top-down, those become parameters with separate jobs:

- `shock_sigma` — how much a stay date's realised demand deviates from `lam_true`.
- `shock_corr_len` — over how many days of DTA that deviation is correlated. This is what decides
  how much of the deviation is *visible early*: a long correlation length means a date that runs
  hot at DTA 90 is still running hot at DTA 30 and will finish high, so the partial curve is
  informative beyond the level. A short one makes the deviation wash out.
- `increment_dispersion` — irreducible per-cell lumpiness, `var/mean` of the booking increment
  around its intensity. Stands in for batch arrivals (a group block landing on one day) without
  simulating batches.

The split is the point. The first two are demand-level and *partly forecastable* from context and
from early pace; the third is noise no model can recover. Piling dispersion into the third raises
every model's error floor equally and compresses the gap between architectures, which makes the
benchmark noisier without making it more discriminating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ArchetypeConfig:
    cap: tuple[int, int]
    base_demand: float
    """Gross demand as a multiple of capacity, before wash and before capacity rejection.

    Renamed from v1's `base_occ`, which was misleading: roughly a fifth of gross demand cancels
    and some is turned away, so a value near 1.0 here lands at ~0.58 realised occupancy. Naming it
    occupancy invited exactly the mistake of reading `lam / capacity` as expected fullness — see
    `GeneratorConstants.expected_wash`.
    """
    adr: tuple[float, float]
    dow: tuple[float, ...]  # Mon..Sun, mean 1.0
    month: tuple[float, ...]  # Jan..Dec, sums to 12.0
    yearly: dict[str, tuple[float, float]]  # {"a": (h1, h2), "b": (h1, h2)}
    hol: dict[str, float]
    events: tuple[int, int]
    event_mult: tuple[float, float]
    month_end_push: float


@dataclass(frozen=True)
class ComponentConfig:
    """One lead-time segment of the booking-pace mixture. `mu` is log-space median DTA."""

    mu: float
    sigma: float


ARCHETYPES: dict[str, ArchetypeConfig] = {
    "urban_business": ArchetypeConfig(
        cap=(120, 320), base_demand=0.936, adr=(180, 320),
        dow=(1.08, 1.22, 1.24, 1.10, 0.80, 0.72, 0.84),
        month=(0.82, 0.95, 1.08, 1.12, 1.10, 1.05, 0.90, 0.88, 1.15, 1.18, 1.02, 0.75),
        yearly=dict(a=(0.05, 0.02), b=(-0.03, 0.01)),
        hol=dict(thanksgiving=0.35, christmas=0.28, july4=0.55, nye=0.60,
                 memorial=0.70, labor=0.70, mlk=0.85),
        events=(1, 3), event_mult=(1.15, 1.45),
        month_end_push=0.04,
    ),
    "resort": ArchetypeConfig(
        cap=(180, 420), base_demand=0.884, adr=(260, 620),
        dow=(0.80, 0.78, 0.80, 0.92, 1.25, 1.38, 1.07),
        month=(0.78, 0.88, 1.05, 1.05, 1.02, 1.20, 1.32, 1.28, 0.88, 0.85, 0.82, 0.87),
        yearly=dict(a=(0.08, 0.03), b=(0.04, -0.02)),
        hol=dict(thanksgiving=1.35, christmas=1.50, july4=1.60, nye=1.70,
                 memorial=1.45, labor=1.40, mlk=1.15),
        events=(0, 2), event_mult=(1.10, 1.30),
        month_end_push=0.0,
    ),
    "airport": ArchetypeConfig(
        cap=(90, 200), base_demand=1.014, adr=(120, 190),
        dow=(1.10, 1.18, 1.18, 1.12, 0.88, 0.76, 0.78),
        month=(0.88, 0.95, 1.04, 1.06, 1.05, 1.08, 1.06, 1.04, 1.05, 1.06, 0.95, 0.78),
        yearly=dict(a=(0.03, 0.01), b=(-0.01, 0.00)),
        hol=dict(thanksgiving=0.80, christmas=0.65, july4=0.85, nye=0.90,
                 memorial=0.90, labor=0.90, mlk=0.95),
        events=(0, 1), event_mult=(1.05, 1.15),
        month_end_push=0.02,
    ),
    "convention": ArchetypeConfig(
        cap=(300, 650), base_demand=0.845, adr=(200, 380),
        dow=(1.05, 1.15, 1.18, 1.10, 0.90, 0.85, 0.77),
        month=(0.85, 1.02, 1.12, 1.10, 1.08, 1.00, 0.85, 0.85, 1.12, 1.20, 1.05, 0.76),
        yearly=dict(a=(0.06, 0.03), b=(-0.02, 0.02)),
        hol=dict(thanksgiving=0.40, christmas=0.30, july4=0.60, nye=0.75,
                 memorial=0.75, labor=0.75, mlk=0.90),
        events=(4, 8), event_mult=(1.45, 2.20),
        month_end_push=0.03,
    ),
    "suburban_select": ArchetypeConfig(
        cap=(80, 160), base_demand=0.910, adr=(110, 165),
        dow=(1.05, 1.12, 1.12, 1.05, 0.92, 0.88, 0.86),
        month=(0.88, 0.96, 1.05, 1.08, 1.08, 1.10, 1.05, 1.04, 1.06, 1.06, 0.92, 0.72),
        yearly=dict(a=(0.04, 0.01), b=(-0.02, 0.01)),
        hol=dict(thanksgiving=0.75, christmas=0.60, july4=0.90, nye=0.85,
                 memorial=0.95, labor=0.95, mlk=1.00),
        events=(0, 2), event_mult=(1.10, 1.25),
        month_end_push=0.02,
    ),
}

# Median DTA 292 / 82 / 12 days, as log-space mu.
#
# Longer than they look, and longer than the v1 generator's 110/32/4.5, for a reason that only
# shows up top-down. v1's BASE_MIX weighted *reservations*; because a group reservation carried
# ~20 rooms and a last-minute one carried ~1.2, group ended up roughly 60% of **rooms** despite
# being 8% of reservations, and the room-weighted curve came out far earlier than the weights
# suggested. Here the pace function allocates rooms directly, so the weights below are room
# shares and the medians have to carry the lead time on their own.
#
# Calibrated so the day-of-week-weighted median booking lead is **69 days**, matching Antonio
# et al. (119,390 real bookings): 64 days for a weekday arrival, 82 for a weekend one.
COMPONENTS: dict[str, ComponentConfig] = {
    "group": ComponentConfig(mu=5.675813, sigma=0.55),
    "leisure": ComponentConfig(mu=4.405103, sigma=0.80),
    "lastmin": ComponentConfig(mu=2.445009, sigma=0.95),
}

# Room shares by segment — (group, leisure, lastmin), rows sum to 1.
#
# **Portfolio-wide, deliberately not per archetype.** Two reasons, and the second is the one that
# matters. First, §3.2 of the spec says curve shape is a consequence of regime and never a knob;
# a hand-set mixture per archetype is exactly such a knob, and it let the archetype reach the
# curve without passing through any mechanism. Second, `w_group/leisure/lastmin` is the target of
# the §7.3 latent-recovery diagnostic — "does the model's learned segment mixture correlate with
# the true one". If `w` were a deterministic function of archetype, that correlation would be
# achievable by memorising archetype, and the diagnostic would prove nothing about whether the
# model reads the surface.
#
# Archetypes still book differently, but now *derivatively*: their demand patterns give them
# different `occ_expected`, which moves the mixture and the compression shift. Pace differences
# are earned through the regime rather than stipulated.
BASE_MIX: dict[str, tuple[float, float, float]] = {
    "weekday": (0.20, 0.48, 0.32),
    "weekend": (0.22, 0.59, 0.19),
}

# the 28-point bucketed DTA grid: daily out to two weeks, then widening
DTA_BUCKETS: tuple[int, ...] = tuple(
    [*range(15), 21, 28, 35, 45, 60, 75, 90, 120, 150, 180, 240, 300, 365]
)


@dataclass(frozen=True)
class GeneratorConstants:
    """Every non-archetype scalar, and every ablation switch.

    Defaults reproduce the shipped generator. An ablation is this dataclass with one field
    neutralised; neutralising one never changes how much randomness a property's stream consumes,
    so paired runs stay comparable.
    """

    # --- booking-pace shape ---
    kappa: float = 0.35  # compression strength; 0.0 ablates the location shift
    book_dow: tuple[float, ...] = (1.18, 1.22, 1.16, 1.08, 0.88, 0.55, 0.63)  # Mon..Sun
    mixture_response: bool = True  # False freezes the weights at BASE_MIX

    # --- the pace shock: demand-level dispersion, and how early it shows ---
    shock_sigma: float = 0.80
    """Stationary sd of the log pace-shock field. 0.0 removes date-level demand deviation."""
    shock_corr_len: float = 240.0
    """Correlation length of that field in days of DTA.

    The forecastability knob. Long means a date's deviation persists across its whole booking
    window, so early pace predicts the final number; short means it decorrelates and the partial
    curve says little beyond the level. Setting it near zero turns demand-level dispersion into
    noise without changing its magnitude, which is the cleanest test of whether a model is using
    pace structure at all.
    """

    # --- irreducible per-cell noise ---
    increment_dispersion: float = 4.0
    """var/mean of the booking increment around its intensity. 1.0 is Poisson counting."""

    # --- promo calendar ---
    n_promos: int = 18
    promo_mult: tuple[float, float] = (1.6, 2.6)
    promo_max_days: int = 3

    # --- demand intensity ---
    noise_sigma: float = 0.11
    mood_phi: float = 0.93
    mood_sigma: float = 0.09
    mood_burn_in: bool = True
    """Start the AR(1) at its stationary variance rather than at a single innovation.

    v1 left this off, so its first ~40 weeks were systematically less variable than the rest — an
    undocumented second trend on top of the recovery ramp. Fixed here.
    """

    # --- capacity ---
    capacity_enforced: bool = True

    # --- cancellations ---
    expected_wash: float = 0.24
    """Planning-time wash assumption, used only to turn `lam` into expected *net* occupancy.

    `lam` is gross demand; roughly a quarter of it cancels. The compression shift and the mixture
    response are supposed to key on how full the date will actually end up, so they read
    `lam * (1 - expected_wash) / capacity` rather than `lam / capacity`. Without the correction,
    any portfolio calibrated to a realistic *net* occupancy sits at gross demand above capacity,
    every date reads as maximally busy, and the whole portfolio compresses — which silently
    breaks the lead-time calibration. Should track the realised wash rate; the band check on wash
    is what catches it drifting.
    """
    cancel_base: float = 0.06
    cancel_amp: float = 0.24
    cancel_tau: float = 70.0
    cancel_group_extra: float = 0.5
    """Extra wash per unit of group weight: p_can *= 1 + cancel_group_extra * w_group."""
    cancel_cap: float = 0.55
    cancel_timing_beta: tuple[float, float] = (1.2, 2.5)

    # --- rate ladder ---
    adr_ladder_base: float = 0.82
    adr_ladder_amp: float = 0.55
    adr_ladder_exp: float = 1.6
    adr_closein_amp: float = 0.10
    adr_closein_tau: float = 6.0

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> GeneratorConstants:
        """Rebuild from a written `config.json`.

        JSON has no tuple, so tuple fields come back as lists; passing those straight to the
        constructor yields an object that hashes differently from the original, which would defeat
        the point of `generator_hash`. Coerce them back.
        """
        types = {f.name: f.type for f in fields(cls)}
        coerced = {
            k: tuple(v) if isinstance(v, list) and "tuple" in str(types.get(k, "")) else v
            for k, v in snapshot.items()
            if k in types
        }
        return cls(**coerced)


def _canonical(obj) -> object:
    """Deterministic, JSON-safe view of a config object — ordering fixed, floats exact."""
    if is_dataclass(obj):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in sorted(fields(obj), key=lambda x: x.name)
        }
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        return repr(obj)
    return obj


def generator_hash(constants: GeneratorConstants) -> str:
    """12-hex digest over ARCHETYPES + COMPONENTS + BASE_MIX + DTA_BUCKETS + constants."""
    payload = json.dumps(
        _canonical({
            "archetypes": ARCHETYPES,
            "components": COMPONENTS,
            "base_mix": BASE_MIX,
            "dta_buckets": DTA_BUCKETS,
            "constants": constants,
        }),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SimConfig:
    seed: int = 20260909
    start: date = date(2023, 1, 1)
    end: date = date(2025, 12, 31)
    k_max: int = 365
    n_props: int = 60
    as_of: date = date(2026, 1, 1)
    dta_grid: tuple[int, ...] = DTA_BUCKETS
    output_dir: Path = Path("_data")
    dataset_version: str = "v3"
    constants: GeneratorConstants = GeneratorConstants()

    @property
    def generator_hash(self) -> str:
        return generator_hash(self.constants)
