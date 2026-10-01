"""Render the T1 parameter-recovery figures (R1-R7) for a dataset.

Usage:
    uv run python -m scripts.plot_recovery --dataset _data/v3
    uv run python -m scripts.plot_recovery --dataset _data/v3 --out reports/figures/recovery

Flags:
    --dataset   dataset directory to load (default: _data/v3)
    --out       figure directory (default: reports/figures/recovery)

Spec: docs/design.md §12 (2026-09-17 entries).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.recovery_figures import (
    fig_calendar_parity,
    fig_lam_residual,
    fig_shape_ratio,
    fig_shock_bands,
    fig_yearly_seasonality,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="_data/v3")
    parser.add_argument("--out", default="reports/figures/recovery")
    args = parser.parse_args()

    ds = LoadedDataset.load(args.dataset)
    out = Path(args.out)

    figures = [
        fig_lam_residual(ds, out / "01_lam_residual.png"),
        fig_shape_ratio(ds, out / "02_shape_ratio.png"),
        fig_shock_bands(ds, out / "03_shock_bands.png"),
        fig_calendar_parity(ds, out / "04_calendar_parity.png"),
        fig_yearly_seasonality(ds, out / "05_yearly_seasonality.png"),
    ]
    for figure in figures:
        print(f"  {figure.path}  — {figure.caption}")


if __name__ == "__main__":
    main()
