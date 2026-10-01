"""Render the (arrival_date, DTA) -> OTB booking-pace surface for one property.

Usage:
    uv run python -m scripts.plot_booking_surface --dataset _data/v100 --property-id 96849
    uv run python -m scripts.plot_booking_surface --dataset _data/v100 --property-id 96849 \\
        --start 2023-01-01 --end 2023-02-28 --dta-max 180

Flags:
    --dataset      dataset directory to load (default: _data/v100)
    --property-id  which property to plot (default: the first row of properties.parquet)
    --start/--end  arrival-date window (default: the dataset's first 90 days)
    --dta-max      largest DTA to show (default: 180)
    --out          output png path (default: _data/figures/<dataset name>/booking_surface.png)
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.surface_figures import fig_booking_surface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="_data/v100")
    parser.add_argument("--property-id", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--dta-max", type=int, default=180)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out) if args.out else Path("_data/figures") / dataset.name / "booking_surface.png"

    ds = LoadedDataset.load(dataset)
    start = args.start
    end = args.end
    if start is None and end is None:
        first = ds.stay_dates["stay_date"].min()
        start, end = str(first.date()), str((first + timedelta(days=59)).date())

    figure = fig_booking_surface(ds, out, property_id=args.property_id,
                                  arrival_start=start, arrival_end=end, dta_max=args.dta_max)
    print(f"  {figure.path}  — {figure.caption}")


if __name__ == "__main__":
    main()
