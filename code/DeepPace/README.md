# DeepPace

**Deep Learning for 2D Hotel Booking-Pace Forecasting.**

Forecast hotel demand by modelling booking history as a 2D booking-pace surface
(`arrival_date` × `DTA`) instead of collapsing it into 1D time-series features.

**Phase I — DeepPace-Sim.** A controlled synthetic hospitality dataset with realistic 2D
booking curves: DOW and seasonality, events, customer-mixture effects, compression,
cancellations, and endogenous ADR. Generated bottom-up from individual reservations with
every latent persisted, so it works as a benchmark and not just as data.

**Phase II — DeepPace-Net.** A deep forecasting architecture that consumes the pace surface
directly, evaluated against 1D baselines that see the same numbers with the geometry removed.

> simulate the booking process → learn its 2D structure → forecast future demand

## Docs


| file                      | what it is                                           |
| ------------------------- | ---------------------------------------------------- |
| `docs/design.md`          | Live engineering spec and decision log — start here |
| `docs/simulation-spec.md` | Normative Phase I dataset spec                       |
| `CLAUDE.md`               | Working conventions for this repo                    |

## Setup

```bash
uv sync                    # base: simulator only, no torch
uv sync --extra net        # + torch and Chronos-2, for Phase II
uv sync --extra eval       # + lightgbm/sklearn baselines
```
