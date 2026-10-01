"""Train, evaluate, reload, and forecast with frozen Chronos-2 plus booking pace.

Usage:
    python scripts/run_net.py --data _data/v3
    python scripts/run_net.py --n-epochs 2 --property-ids 73039

Requires an existing full-grid simulator dataset. Chronos weights are downloaded on
the first uncached forecast; subsequent runs reuse the forecast and training caches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(project_root / "_data/v3"))
    parser.add_argument("--out", default=str(project_root / "_runs/two_stage"))
    parser.add_argument("--as-of", default=None, help="Forecast origin; defaults to --test-start")
    parser.add_argument("--property-ids", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int, default=[7, 10, 30, 90, 180])
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--origin-stride", type=int, default=30)
    parser.add_argument("--shared-days", type=int, default=90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chronos-device", default="cpu")
    parser.add_argument("--cache-dir", default=str(project_root / "_runs/chronos_cache"))
    parser.add_argument("--context-days", type=int, default=730)
    parser.add_argument("--min-history", type=int, default=180)
    parser.add_argument("--train-start", default="2023-07-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument(
        "--exclude-test-origins", nargs="+", default=[], metavar="DATE",
        help="Skip specific evaluation origins, e.g. 2025-12-27",
    )
    args = parser.parse_args()

    import pandas as pd

    from deeppace_net.foundation import ChronosForecaster, FoundationConfig
    from deeppace_net.pace_data import PaceData
    from deeppace_net.train import TrainConfig, load_model, predict, train
    from deeppace_sim.dataset import LoadedDataset

    as_of = pd.Timestamp(args.as_of or args.test_start)
    if as_of <= pd.Timestamp(args.train_end):
        parser.error("--as-of must follow --train-end to avoid training-label leakage")
    foundation = FoundationConfig(
        device=args.chronos_device,
        cache_dir=args.cache_dir,
        context_days=args.context_days,
        min_history=args.min_history,
    )
    config = TrainConfig(
        out_dir=args.out,
        property_ids=args.property_ids,
        horizons=tuple(args.horizons),
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        origin_stride=args.origin_stride,
        shared_days=args.shared_days,
        device=args.device,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        exclude_test_origins=tuple(args.exclude_test_origins),
        foundation=foundation,
    )
    print(f"Loading full-grid dataset: {args.data}", flush=True)
    dataset = LoadedDataset.load(args.data, full_grid=True)
    data = PaceData(dataset, config.property_ids)
    # Fail before training if the requested inference surface is unavailable.
    data.build(as_of, config.horizons, config.shared_days, labeled=False)
    forecaster = ChronosForecaster(foundation)
    print("Training and evaluating (Chronos estimates are generated automatically)...", flush=True)
    result = train(dataset, config, forecaster=forecaster)
    run_dir = Path(result.run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    model = load_model(checkpoint, device=args.device)
    forecasts = predict(
        data, model, forecaster, str(as_of.date()), config.horizons, config.shared_days, include_bare=True
    )
    output = run_dir / f"forecast_{as_of.date()}.csv"
    forecasts.to_csv(output, index=False)
    bare_wmape = next(m["value"] for m in result.bare_metrics
                      if m["metric"] == "wmape" and m["segment"] == "all")
    print(f"Bare Chronos test WMAPE: {bare_wmape:.4f}")
    print(f"Chronos + calendar test WMAPE: {result.metric('wmape', baseline=True):.4f}")
    print(f"Two-stage test WMAPE:   {result.metric('wmape'):.4f}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test predictions and metrics: {run_dir}")
    print(f"As-of forecasts: {output}")
    print(forecasts.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
