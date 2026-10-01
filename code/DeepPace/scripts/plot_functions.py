"""Plot the generator's demand and booking-pace functions.

Usage:
    uv run python -m scripts.plot_functions
    uv run python -m scripts.plot_functions --out reports/figures

Flags:
    --out   figure directory (default: reports/figures)

Needs no dataset: the top-down generator's model *is* a set of functions, so it can be inspected
directly rather than sampled and measured back.

Spec: docs/simulation-spec.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeppace_sim.figures import fig_demand_function, fig_pace_function, fig_shock_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/figures")
    args = parser.parse_args()
    out = Path(args.out)

    figures = [
        fig_demand_function(out / "01_demand_function.png"),
        fig_pace_function(out / "02_pace_function.png"),
        fig_shock_field(out / "03_shock_field.png"),
    ]
    for figure in figures:
        print(f"  {figure.path}  — {figure.caption}")


if __name__ == "__main__":
    main()
