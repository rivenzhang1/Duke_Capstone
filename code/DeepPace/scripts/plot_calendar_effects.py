"""Visualize the booking-date calendar effects recovered from a generated dataset.

Usage:
    uv run python -m scripts.plot_calendar_effects --dataset _data/v100
    uv run python -m scripts.plot_calendar_effects --dataset _data/v100 --out _data/figures/v100

Flags:
    --dataset   dataset directory to load (default: _data/v100)
    --out       figure directory (default: _data/figures/<dataset name>)
    --top-n     how many anomalous booking dates to flag on the residual panel (default: 15)

Detection is blind — it never reads `promo_calendar.parquet` — so it works the same way on a
dataset that predates that file. See `deeppace_sim.calendar_figures` for the method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeppace_sim.calendar_figures import fig_calendar_effects
from deeppace_sim.dataset import LoadedDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="_data/v100")
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out) if args.out else Path("_data/figures") / dataset.name

    ds = LoadedDataset.load(dataset)
    figure = fig_calendar_effects(ds, out / "calendar_effects.png", top_n=args.top_n)
    print(f"  {figure.path}  — {figure.caption}")


if __name__ == "__main__":
    main()
