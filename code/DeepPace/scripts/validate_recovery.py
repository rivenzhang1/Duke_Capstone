"""Check whether lam, shape and shock are implemented as designed (T1 parameter recovery).

Usage:
    uv run python -m scripts.validate_recovery --dataset _data/v3

Runs `recovery.recover_lam/recover_shape/recover_shock` and prints each report: which checks
passed, the measurements behind them, and any notes explaining an expected (not a bug) deviation.

Spec: docs/design.md §12 (2026-09-17 entry); the composition under test is `pace.py`'s
`nu = lam * shape * shock`.
"""

from __future__ import annotations

import argparse

from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.recovery import (
    recover_dow,
    recover_events,
    recover_holidays,
    recover_lam,
    recover_seasonality,
    recover_shape,
    recover_shock,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="_data/v3")
    args = parser.parse_args()

    ds = LoadedDataset.load(args.dataset)
    reports = [
        recover_lam(ds), recover_shape(ds), recover_shock(ds),
        recover_dow(ds), recover_holidays(ds), recover_events(ds), recover_seasonality(ds),
    ]

    all_passed = True
    for report in reports:
        print(f"[{report.tier}] {report.name}: {sum(report.checks.values())}/{len(report.checks)}")
        for name, ok in report.checks.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")
            all_passed &= ok
        for name, value in report.measurements.items():
            print(f"    {name} = {value:.4f}")
        for name, why in report.skipped.items():
            print(f"  [skip] {name} — {why}")
        for name, note in report.notes.items():
            print(f"  note: {name} — {note}")
        print()

    print("all checks passed" if all_passed else "some checks failed")


if __name__ == "__main__":
    main()
