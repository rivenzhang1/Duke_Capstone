"""Evaluate bare, calendar, DTA-covariate and two-stage Chronos cases with validation-only model selection.

Usage:
    python scripts/evaluate_net.py --data _data/v3
    uv run --extra net python -m scripts.evaluate_net --data _data/v3
    uv run --extra net python -m scripts.evaluate_net --property-ids 73039 --epochs 2

Every split end is a label cutoff; every split start is a forecast origin.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from deeppace_eval.benchmark import EvaluationConfig, evaluate
    from deeppace_net.foundation import FoundationConfig
    from deeppace_sim.dataset import LoadedDataset

    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(project_root / "_data/v3"))
    parser.add_argument("--out", default=str(project_root / defaults.out_dir))
    parser.add_argument("--property-ids", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(defaults.horizons))
    parser.add_argument(
        "--compact-grid",
        action="append",
        help="Comma-separated DTA grid; repeat for validation selection",
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=list(defaults.epochs))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=list(defaults.hidden_dims))
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--chronos-device", default=defaults.foundation.device)
    parser.add_argument("--context-days", type=int, default=730)
    parser.add_argument("--min-history", type=int, default=180)
    parser.add_argument("--cache-dir", default=str(project_root / "_runs/chronos_cache"))
    parser.add_argument("--no-shared-ablation", action="store_true")

    # ARIMAX calendar (keep your current default as-is)
    parser.add_argument(
        "--include-arimax-calendar",
        dest="include_arimax_calendar",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-arimax-calendar",
        dest="include_arimax_calendar",
        action="store_false",
    )

    # ARIMAX pace (DEFAULT OFF)
    parser.add_argument(
        "--include-arimax-pace",
        dest="include_arimax_pace",
        action="store_true",
        default=False,
        help="Enable ARIMAX with pickup 1/3/7/14 exogenous variables (default: disabled).",
    )
    parser.add_argument(
        "--no-arimax-pace",
        dest="include_arimax_pace",
        action="store_false",
    )

    parser.add_argument("--arimax-order", default="1,0,1")
    parser.add_argument("--arimax-seasonal-order", default="0,0,0,0")

    for field in (
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ):
        parser.add_argument("--" + field.replace("_", "-"), default=getattr(defaults, field))
    for field in (
        "origin_stride",
        "train_stride",
        "bootstrap_draws",
        "block_days",
        "seed",
        "shared_days",
    ):
        parser.add_argument(
            "--" + field.replace("_", "-"), type=int, default=getattr(defaults, field)
        )
    args = parser.parse_args()

    cfg = replace(
        defaults,
        **{
            field: getattr(args, field)
            for field in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
                "origin_stride",
                "train_stride",
                "bootstrap_draws",
                "block_days",
                "seed",
                "shared_days",
                "device",
            )
        },
        property_ids=tuple(args.property_ids) if args.property_ids else None,
        horizons=tuple(args.horizons),
        compact_grids=grids,
        epochs=tuple(args.epochs),
        hidden_dims=tuple(args.hidden_dims),
        include_shared=not args.no_shared_ablation,
        include_arimax_calendar=args.include_arimax_calendar,
        include_arimax_pace=args.include_arimax_pace,
        arimax_order=tuple(int(x) for x in args.arimax_order.split(",")),
        arimax_seasonal_order=tuple(int(x) for x in args.arimax_seasonal_order.split(",")),
        out_dir=args.out,
        foundation=FoundationConfig(
            device=args.chronos_device,
            context_days=args.context_days,
            min_history=args.min_history,
            cache_dir=args.cache_dir,
        ),
    )

    dataset = LoadedDataset.load(
        Path(args.data),
        property_ids=tuple(args.property_ids) if args.property_ids else None,
        full_grid=True,
    )
    evaluate(dataset, cfg)
    print(
        f"Cases: {', '.join(result.cases)}; {result.n_test_rows_per_case} paired test rows per case"
    )
    print(f"Report, predictions, model selection and comparisons: {result.run_dir}")


if __name__ == "__main__":
    main()
