"""DeepPace-Sim — a top-down generator for 2D hotel booking-pace surfaces.

The booking curve is produced by composing a **demand function** with a **pace function**, not by
aggregating simulated reservations:

    nu(p, d, k) = lam(p, d) * shape(p, d, k) * shock(p, d, k)

`lam` is the stay date's demand level, `shape` its systematic booking curve, `shock` its own
deviation. Integer increments, cancellations and capacity rejection turn that intensity into an
observed surface.

The predecessor generated individual reservations as a Poisson process; `deeppace_sim_v1/README.md`
records what that cost and why this replaced it. In short, two properties a forecasting benchmark
needs to control were consequences of the party-size distribution rather than choices: how
dispersed demand is, and how much of a date's deviation is visible early.

Modules:
    config      archetypes, pace components, GeneratorConstants, SimConfig, generator_hash
    calendars   holidays, per-property events, portfolio-wide promo diagonals
    demand      lam(p, d) — multiplicative seasonality, AR(1) mood, trend
    pace        the booking-pace function: mixture, compression, calendar, shock field
    curve       increments, cancellations, capacity walk -> the observed curve
    generate    portfolio driver composing the above
    writer      parquet emit + config snapshot; owns the on-disk schema
    dataset     LoadedDataset — the one reader every check shares
    validate    identities and plausibility bands
"""

from deeppace_sim.config import ARCHETYPES, COMPONENTS, GeneratorConstants, SimConfig

__all__ = ["ARCHETYPES", "COMPONENTS", "GeneratorConstants", "SimConfig"]
