"""Training loop for DeepPace-Net.

Design: docs/design.md §4.5, §7.2, §12 (2026-09-18 dynamic-factor entry).

Splits: train 2023-2024, validate/test 2025. That boundary contains a genuine level shift
(post-COVID recovery ramp into ~3%/yr growth). It is realistic; do not smooth it away.

**Target is `unconstrained_demand`, not `rooms_sold`.** Capacity/cancellation are out of scope
for `dynamic_factor` (docs/design.md §12): no censored likelihood, no capacity term in the loss.
A `min(prediction, capacity)` cap is available as a strictly separate post-processing step
(`cap_at_capacity`), never inside training.

Checkpoints and run configs go to ``_runs/<variant>-<version>/``; each result file records
seed, dataset version, and variant version.

Public surface:
    TrainConfig (dataclass), TrainResult (dataclass)
    train(variant, ds, config) -> TrainResult
    cap_at_capacity(predictions, stream, ds) -> np.ndarray
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deeppace_eval.metrics import score
from deeppace_net import registry
from deeppace_net.booking_stream import BookingStream, build_booking_stream
from deeppace_sim.dataset import LoadedDataset


@dataclass(frozen=True)
class TrainConfig:
    variant: str = "dynamic_factor"
    hidden_dim: int = 16
    n_epochs: int = 20
    lr: float = 5e-3
    weight_decay: float = 1e-4
    chunk_days: int = 30
    property_ids: list[int] | None = None
    train_start: str = "2023-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    seed: int = 20260826
    device: str = "cpu"


HORIZONS = (7, 30, 90, 180)


@dataclass
class TrainResult:
    variant: str
    version: int
    config: dict
    train_loss_by_epoch: list[float] = field(default_factory=list)
    test_metrics: list[dict] = field(default_factory=list)
    baseline_metrics: list[dict] = field(default_factory=list)
    n_train_slots: int = 0
    n_test_slots: int = 0

    def metric(self, name: str, segment: str = "all", baseline: bool = False) -> float:
        rows = self.baseline_metrics if baseline else self.test_metrics
        for row in rows:
            if row["metric"] == name and row["segment"] == segment:
                return row["value"]
        raise KeyError(f"no {name!r} metric for segment {segment!r}")

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def train(ds: LoadedDataset, config: TrainConfig = TrainConfig()) -> TrainResult:
    torch.manual_seed(config.seed)
    module = registry.get_module(config.variant)
    build = module.build

    train_stream = build_booking_stream(
        ds, start=config.train_start, end=config.train_end, property_ids=config.property_ids,
    )
    # as_of is the train/test boundary, not test_end: every 2025 arrival date is forecast from
    # the same vantage point (end of training), at whatever horizon (arrival_date - as_of) that
    # date happens to sit at — a genuine, leakage-safe forecast, not a look-ahead.
    test_stream = build_booking_stream(
        ds, start=config.test_start, end=config.test_end, property_ids=config.property_ids,
        as_of=config.train_end,
    )

    model_config = module.DynamicFactorConfig(
        n_slot_features=train_stream.slot_context.shape[1], hidden_dim=config.hidden_dim,
    )
    spec = build(model_config)
    model = spec.model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    result = TrainResult(
        variant=spec.name, version=spec.version, config=asdict(config),
        n_train_slots=len(train_stream.slots), n_test_slots=len(test_stream.slots),
    )

    for epoch in range(config.n_epochs):
        model.train()
        optimizer.zero_grad()
        preds_log, targets_log, _ = model(train_stream, chunk_days=config.chunk_days, device=config.device)
        loss = torch.nn.functional.mse_loss(preds_log, targets_log)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        result.train_loss_by_epoch.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        preds_log, _, slot_idx = model(test_stream, chunk_days=config.chunk_days, device=config.device)
        pred = np.expm1(preds_log.cpu().numpy()) * test_stream.capacity[slot_idx]
        target = test_stream.target[slot_idx]

        # score only the *final* (largest-booking_date-consumed) prediction per slot — the
        # forecast made from the as-of boundary itself, comparable to a real forecast at test
        # time — and at whatever horizon (arrival_date - as_of) that slot happens to sit at
        final_pred, final_target, final_slots = _final_touch(pred, target, slot_idx, len(test_stream.slots))
        naive_pred, naive_target, naive_slots = _naive_baseline(test_stream)

        as_of = pd.Timestamp(config.train_end)
        horizon = (test_stream.slots["stay_date"] - as_of).dt.days.to_numpy()

        result.test_metrics = [
            asdict(m) for m in score(final_target, final_pred, horizon=horizon[final_slots], horizons=HORIZONS)
        ]
        result.baseline_metrics = [
            asdict(m) for m in score(naive_target, naive_pred, horizon=horizon[naive_slots], horizons=HORIZONS)
        ]

    return result


def _final_touch(
    pred: np.ndarray, target: np.ndarray, slot_idx: np.ndarray, n_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The last (pred, target) pair for each slot, in booking-date order — one row per slot.

    `slot_idx` is already in ascending booking-date order (the order `forward` touched them),
    so the last occurrence of each slot index is exactly its as-of-boundary prediction.
    """
    last_pred = np.full(n_slots, np.nan)
    last_target = np.full(n_slots, np.nan)
    last_pred[slot_idx] = pred
    last_target[slot_idx] = target
    mask = ~np.isnan(last_pred)
    return last_pred[mask], last_target[mask], np.nonzero(mask)[0]


def _naive_baseline(stream: BookingStream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`net_so_far` at the last touch, with no learned correction — bookkeeping alone."""
    n_slots = len(stream.slots)
    net_so_far = np.zeros(n_slots)
    last_net = np.full(n_slots, np.nan)
    touched = np.zeros(n_slots, dtype=bool)
    for day in stream.days:
        idx = day.slot_idx
        net_so_far[idx] += day.gross_inc - day.cancel_inc
        last_net[idx] = net_so_far[idx]
        touched[idx] = True
    return last_net[touched], stream.target[touched], np.nonzero(touched)[0]


def cap_at_capacity(predictions: np.ndarray, slot_property_ids: np.ndarray, ds: LoadedDataset) -> np.ndarray:
    """Post-processing only — never used inside the loss. `min(prediction, capacity)`."""
    capacity = ds.properties.set_index("property_id")["capacity"].reindex(slot_property_ids).to_numpy()
    return np.minimum(predictions, capacity)
