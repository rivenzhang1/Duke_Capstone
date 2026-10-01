"""Encoder variants — one module per architecture.

The default is `two_stage`: frozen Chronos-2 plus an observed-pace adjustment to final rooms
sold. Other candidates remain research comparisons, not established accuracy improvements.
Candidates:

    conv2d          2D conv / U-net over the masked surface; cheap, sees diagonals natively
    axial           attention along DTA (pace) and along arrival_date (seasonality) separately
    hier_state      summarise each arrival date's partial curve into a context-conditioned state
                    vector, then model the sequence of states
    diagonal        explicit reindexing to (booking_date, dta) as an auxiliary view
    dynamic_factor  IMPLEMENTED (design.md §12, 2026-09-18) — shared/local GRU state-space,
                    walked in booking-date order; see its own module docstring. Beats a
                    bookkeeping-only baseline at DTA>90 on v3, loses at DTA<30. Not yet compared
                    against the other four or against §6's flat-GBM baseline.

Pick against the flattened gradient-boosting baseline (§6), not against naive pickup — the
GBM sees the same cells with the geometry destroyed, so it is the actual hypothesis test.
"""

from __future__ import annotations

__all__: list[str] = ["dynamic_factor", "two_stage"]
