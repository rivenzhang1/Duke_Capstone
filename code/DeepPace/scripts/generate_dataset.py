"""Generate the DeepPace-Sim dataset.

Usage:
    uv run python -m scripts.generate_dataset --out _data/v3
    uv run python -m scripts.generate_dataset --out _data/dev --limit 5 --bucketed-only

Flags:
    --out             output directory (default: _data/<dataset_version>)
    --seed            RNG seed
    --limit N         first N properties only, for iteration
    --bucketed-only   skip the full-grid on_books.parquet
    --validate        run the identity checks and anchors after generation

Spec: docs/simulation-spec.md.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from deeppace_sim.config import SimConfig
from deeppace_sim.generate import simulate_portfolio
from deeppace_sim.writer import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bucketed-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    config = SimConfig(seed=args.seed)
    if args.limit is not None:
        config = dataclasses.replace(config, n_props=args.limit)
    out_dir = Path(args.out) if args.out else config.output_dir / config.dataset_version

    paths = write_dataset(
        list(simulate_portfolio(config)), config, out_dir, bucketed_only=args.bucketed_only
    )
    print(f"wrote {out_dir}  (generator {config.generator_hash})")
    for name, path in vars(paths).items():
        if path:
            print(f"  {name + ':':<20} {path}")

    if args.validate:
        from deeppace_sim.dataset import LoadedDataset
        from deeppace_sim.validate import anchors, validate_dataset

        ds = LoadedDataset.load(out_dir)
        for report in (validate_dataset(ds), anchors(ds)):
            print(f"\n[{report.tier}] {report.name}: "
                  f"{sum(report.checks.values())}/{len(report.checks)}")
            for name, ok in report.checks.items():
                print(f"  [{'OK' if ok else 'FAIL'}] {name}")
            for name, value in report.measurements.items():
                print(f"  {name} = {value:.4f}")
            for name, why in report.skipped.items():
                print(f"  [skip] {name} — {why}")


if __name__ == "__main__":
    main()
