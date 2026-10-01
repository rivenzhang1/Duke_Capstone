"""Shared evaluation for both phases.

Kept as one package so Phase I plausibility checks and Phase II scoring cannot drift apart.

Public modules (see docs/four-case-evaluation.md):
    benchmark  four-case rolling-origin evaluation and validation-only selection
    covariates Chronos history/full-pace/compact-pace as-of input construction
    comparison aligned scores and paired origin-block bootstrap intervals
    profiling  wall time and sampled process RSS for cost accounting
    metrics    Bias, MAPE, WMAPE — per horizon and per segment
    baselines  the 1D baselines DeepPace-Net must beat
    latents    ground-truth recovery diagnostics the simulator uniquely allows
"""

from __future__ import annotations

__all__ = ["metrics", "baselines", "latents", "benchmark", "covariates", "comparison", "profiling"]
