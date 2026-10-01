# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**DeepPace: Deep Learning for 2D Hotel Booking-Pace Forecasting.**

Forecast hotel demand by treating booking history as what it actually is — a **2D
booking-pace surface** indexed by `(arrival_date, DTA)` — instead of collapsing it into
conventional 1D time-series features.

The research narrative runs in one line: **simulate the booking process → learn its 2D
structure → forecast future demand.** The two phases below are that line split in half.

### Phase I — DeepPace-Sim

Develop a **controlled synthetic hospitality dataset** with realistic 2D booking curves.
In scope: DOW and seasonality, events, customer-mixture effects, compression,
cancellations, and endogenous ADR.

The point of a simulator is that **you know the answer.** Generation is bottom-up from
individual reservations, capacity is enforced by chronological rejection, and every latent
(true intensity, true mixture weights, true arrival pdf, turn-aways, promo calendar) is
persisted. That is what makes the output a *benchmark* rather than merely data.

Full spec: `docs/simulation-spec.md`.

### Phase II — DeepPace-Net

Develop a **deep neural forecasting architecture** that consumes the booking-pace surface
directly. The design axis under test is whether preserving the `(arrival_date, DTA)`
geometry — and the diagonal `booking_date` structure that lives inside it — beats a strong
1D baseline that sees the same information as flattened features.

Why the 2D framing is the hypothesis and not a detail:

- **The two axes carry different physics.** Movement along `arrival_date` is seasonality and
  demand level; movement along `DTA` is *pace* — how a given date is filling. A model that
  flattens the surface has to rediscover that separation from scratch.
- **The same on-books number means opposite things in different contexts.** At long DTA a
  peak date and a trough date can both sit at low on-books; only the surrounding context
  disambiguates them. This is the pathology the simulator is built to reproduce (see
  `docs/simulation-spec.md` §5.2).
- **Calendar effects are diagonal.** `booking_date = arrival_date − DTA`, so booking-DOW
  ripples and flash-sale promos appear as *diagonal streaks* in the surface. They are
  invisible to any model whose features are built per `(arrival_date, DTA)` cell
  independently.

### Success criterion

Accuracy against 1D baselines on the simulated data (Bias, MAPE, WMAPE across horizons),
**plus** the latent-recovery checks the simulator uniquely enables — chiefly, whether the
model's learned segment mixture correlates with the ground-truth `w_group/leisure/lastmin`.
A pure accuracy win with zero latent correlation means the 2D structure is not doing the
work we claim.

## The design doc is the live spec — keep it current

`docs/design.md` is the engineering spec **and** the live progress/decision tracker for this
repo (build tracker in its §8, decisions log in its §12). On any **major** change — a new or removed
module, a variant `VERSION` bump, a contract or algorithm change, a new default, or a change to eval
methodology — update the affected design-doc section **in the same change**, and note it in §12.
Prefer **merging** a durable change up into the body (so the spec always reads as current truth) over
letting the notes log accumulate contradictions; **remove** log entries for aborted or superseded
experiments — nothing in §12 should be obviously inconsistent with the body above it. A behavior
change landed without a corresponding design-doc update is incomplete.

`docs/simulation-spec.md` is under the same rule for Phase I: if the generator's behavior
changes, the spec changes in the same edit.

## Tooling

- **Package manager: `uv`.** Run everything through it — `uv run python -m <pkg>.<script>`.
  Add deps with `uv add`; never invoke a bare `python`/`pip`.
- **Python ≥ 3.12.** Use `from __future__ import annotations` and modern typing
  (`X | None`, `list[str]`).
- Pin platform-sensitive wheels (e.g. CUDA torch) explicitly in `pyproject.toml` under
  `[tool.uv.sources]` so they can't silently downgrade.

## Naming: `_` prefix = temporary / local-only

Files and directories whose name starts with `_` are scratch, drafts, or local-only
artifacts, excluded from version control via `.gitignore` (rule: `_*`). Use it for scratch
scripts, local fixtures/harnesses, and generated intermediates (e.g. `_data/`, `_runs/`).
Do NOT use `_` for source code that should be versioned. Whitelisted exceptions:
`__init__.py`, `__main__.py`; add any other explicitly and prefer renaming to drop the
underscore.

**Generated parquet is never committed.** The simulated dataset is ~400 MB at the full DTA
grid; it lives in `_data/` and is reproduced from seed + config.
