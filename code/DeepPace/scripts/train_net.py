"""Train Chronos-2 plus an observed-pace adjustment to final rooms sold.

Usage:
    uv run --extra net python -m scripts.train_net --data _data/v3
    uv run --extra net python -m scripts.train_net --property-ids 73039 --n-epochs 2
"""

from __future__ import annotations

import argparse

from deeppace_net.foundation import FoundationConfig
from deeppace_net.train import TrainConfig, train
from deeppace_sim.dataset import LoadedDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="_data/v3")
    parser.add_argument("--out", default="_runs/two_stage")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chronos-device", default="cpu")
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--chronos-revision", default=FoundationConfig().revision)
    parser.add_argument("--context-days", type=int, default=730)
    parser.add_argument("--min-history", type=int, default=180)
    parser.add_argument("--origin-stride", type=int, default=30)
    parser.add_argument("--shared-days", type=int, default=90)
    parser.add_argument("--horizons", nargs="+", type=int, default=[7, 10, 30, 90, 180])
    parser.add_argument("--property-ids", nargs="+", type=int)
    for name, default in [
        ("train-start", "2023-07-01"),
        ("train-end", "2024-12-31"),
        ("test-start", "2025-01-01"),
        ("test-end", "2025-12-31"),
    ]:
        parser.add_argument(f"--{name}", default=default)
    args = parser.parse_args()
    config = TrainConfig(
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        property_ids=args.property_ids,
        out_dir=args.out,
        origin_stride=args.origin_stride,
        shared_days=args.shared_days,
        horizons=tuple(args.horizons),
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        foundation=FoundationConfig(
            model_id=args.chronos_model,
            revision=args.chronos_revision,
            device=args.chronos_device,
            context_days=args.context_days,
            min_history=args.min_history,
        ),
    )
    result = train(LoadedDataset.load(args.data), config)
    print(f"Two-stage WMAPE: {result.metric('wmape'):.4f}")
    print(f"Chronos-only WMAPE: {result.metric('wmape', baseline=True):.4f}")
    print(f"Saved checkpoint, predictions and metrics: {result.run_dir}")


if __name__ == "__main__":
    main()
