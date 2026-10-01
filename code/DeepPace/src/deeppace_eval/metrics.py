"""Forecast accuracy metrics.

Design: docs/design.md §7.1.

Bias, MAPE, WMAPE, reported **per horizon (7 / 30 / 90 / 180 days out)** and broken out by
archetype and by sell-out status. Portfolio aggregates hide exactly the failures that matter:
peak dates and censored dates. A single headline number is not an acceptable report.

Public surface:
    MetricResult (dataclass) — metric name, value, n, segment key
    score(y_true, y_pred, horizon=None, horizons=(7,30,90,180), segment=None) -> list[MetricResult]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    metric: str
    value: float
    n: int
    segment: str = "all"


def _score_segment(y_true: np.ndarray, y_pred: np.ndarray, segment: str) -> list[MetricResult]:
    n = len(y_true)
    bias = float(np.mean(y_pred - y_true))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero]))) if nonzero.any() else float("nan")
    wmape = float(np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true))) if n else float("nan")
    return [
        MetricResult("bias", bias, n, segment),
        MetricResult("mape", mape, n, segment),
        MetricResult("wmape", wmape, n, segment),
    ]


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: np.ndarray | None = None,
    horizons: tuple[int, ...] = (7, 30, 90, 180),
    segment: np.ndarray | None = None,
) -> list[MetricResult]:
    """Bias/MAPE/WMAPE overall, plus per-horizon-bucket and per-segment breakdowns.

    `horizon` (e.g. arrival_date - as_of, in days) is bucketed into `(0, h0], (h0, h1], ...,
    (h_last, inf)`. `segment` is any per-example label (archetype, sold_out, ...); one triplet
    of metrics is reported per distinct value.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    results = _score_segment(y_true, y_pred, "all")

    if horizon is not None:
        horizon = np.asarray(horizon)
        edges = (0, *horizons, np.inf)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (horizon > lo) & (horizon <= hi)
            if not mask.any():
                continue
            label = f"horizon_{int(lo)}-{int(hi) if np.isfinite(hi) else 'inf'}"
            results += _score_segment(y_true[mask], y_pred[mask], label)

    if segment is not None:
        segment = np.asarray(segment)
        for value in np.unique(segment):
            mask = segment == value
            results += _score_segment(y_true[mask], y_pred[mask], f"segment_{value}")

    return results
