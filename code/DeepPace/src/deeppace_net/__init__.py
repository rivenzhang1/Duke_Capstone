"""DeepPace-Net (Phase II) — deep forecasting over the 2D booking-pace surface.

Consumes the ``(arrival_date, DTA)`` surface directly rather than collapsing it into 1D
features. **torch lives in this package and only in this package** — generating the Phase I
dataset must never pull it in.

Design: ``docs/design.md`` §4-§5. The encoder architecture is an open question (§11); the
registry exists so variants can be compared rather than chosen up front.

Public modules (see docs/two-stage-forecasting.md):
    foundation frozen Chronos-2 forecasts from final rooms-sold history
    pace_data  origin-based partial curves and booking-date summaries
    windows   as-of slicing into masked surface tensors (leakage-tested)
    features  calendar and property covariates aligned to the surface
    registry  NAME/VERSION/entry-fn variant registry
    models/   one module per encoder variant
    train     training loop and checkpointing
"""

from __future__ import annotations

__all__ = ["windows", "features", "registry", "models", "train", "foundation", "pace_data"]
