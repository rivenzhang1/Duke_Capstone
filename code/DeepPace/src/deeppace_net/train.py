"""Rolling-origin training of frozen Chronos plus a final rooms-sold pace adjustment.

Usage:
    uv run --extra net python -m scripts.train_net --data _data/v3
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deeppace_eval.metrics import score
from deeppace_net.foundation import ChronosForecaster, FoundationConfig
from deeppace_net.models.two_stage import (
    NAME,
    VERSION,
    TwoStageConfig,
    TwoStagePaceNet,
    retention_penalty,
)
from deeppace_net.pace_data import PaceBatch, PaceData
from deeppace_sim.dataset import LoadedDataset


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 16
    n_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_retention: float = 0.5  # see models.two_stage.retention_penalty
    retention_weight: float = 0.2
    property_ids: list[int] | None = None
    train_start: str = "2023-07-01"  # first forecast origin
    train_end: str = "2024-12-31"  # last permitted training label
    test_start: str = "2025-01-01"  # first test origin
    test_end: str = "2025-12-31"  # last permitted test label
    origin_stride: int = 30
    horizons: tuple[int, ...] = (1, 7, 10, 30, 90, 180)
    shared_days: int = 90
    seed: int = 20260826
    device: str = "cpu"
    out_dir: str = "_runs/two_stage"
    foundation: FoundationConfig = field(default_factory=FoundationConfig)


@dataclass
class TrainResult:
    variant: str
    version: int
    config: dict
    train_loss_by_epoch: list[float] = field(default_factory=list)
    test_metrics: list[dict] = field(default_factory=list)
    baseline_metrics: list[dict] = field(default_factory=list)
    otb_metrics: list[dict] = field(default_factory=list)
    n_train_slots: int = 0
    n_test_slots: int = 0
    run_dir: str = ""

    def metric(self, name: str, segment: str = "all", baseline: bool = False) -> float:
        rows = self.baseline_metrics if baseline else self.test_metrics
        return next(m["value"] for m in rows if m["metric"] == name and m["segment"] == segment)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def base_forecast(data: PaceData, batch: PaceBatch, forecaster: ChronosForecaster) -> np.ndarray:
    t = pd.Timestamp(batch.rows.as_of.iloc[0])
    base = np.empty(len(batch.rows), dtype=np.float32)
    for pid, rows in batch.rows.groupby("property_id"):
        history = data.history[(data.history.property_id == pid) & (data.history.stay_date <= t)]
        pred = forecaster.predict(history, t, int(pid), int(rows.horizon.max()))
        base[rows.index] = pred[rows.horizon.to_numpy(dtype=int) - 1]
    return base


def predict(
    data: PaceData,
    model: TwoStagePaceNet,
    forecaster: ChronosForecaster,
    as_of: str,
    horizons: tuple[int, ...],
    shared_days: int = 90,
) -> pd.DataFrame:
    """Inference reads no future labels or simulator latents. Return point forecasts in rooms."""
    batch = data.build(as_of, horizons, shared_days, labeled=False)
    base = base_forecast(data, batch, forecaster)
    model.eval()
    with torch.no_grad():
        raw = model(batch, torch.from_numpy(base)).cpu().numpy()
    result = batch.rows.copy()
    result["base_raw"] = base
    result["base"] = np.clip(base, 0, result.capacity)
    result["prediction_raw"] = raw
    result["prediction"] = np.clip(raw, 0, result.capacity)
    result["adjustment"] = result.prediction - result.base
    result["remaining_net_pickup"] = result.prediction - result.current_otb
    return result


def load_model(checkpoint: str | Path, device: str = "cpu") -> TwoStagePaceNet:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TwoStagePaceNet(TwoStageConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def _metrics(rows: pd.DataFrame, column: str) -> list[dict]:
    results = [asdict(m) for m in score(rows.target, rows[column])]
    for key in ("horizon", "archetype"):
        for value, part in rows.groupby(key):
            results.extend(
                {**asdict(m), "segment": f"{key}_{value}"} for m in score(part.target, part[column])
            )
    return results


def train(
    ds: LoadedDataset,
    config: TrainConfig | None = None,
    forecaster: ChronosForecaster | None = None,
) -> TrainResult:
    config = config or TrainConfig()
    if config.n_epochs < 1 or config.origin_stride < 1:
        raise ValueError("n_epochs and origin_stride must be positive")
    if pd.Timestamp(config.test_start) <= pd.Timestamp(config.train_end):
        raise ValueError("Test origins must follow the training label cutoff")
    torch.manual_seed(config.seed)
    data = PaceData(ds, config.property_ids)
    forecaster = forecaster or ChronosForecaster(config.foundation)
    digest = hashlib.sha256()
    for frame in (data.history, data.properties, data.surface):
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    metadata = {
        "config": asdict(config),
        "variant": NAME,
        "version": VERSION,
        "dataset_version": ds.dataset_version,
        "generator_hash": ds.generator_hash,
        "data_hash": digest.hexdigest(),
        "foundation": forecaster.identity,
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(config.out_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists() and (root / "checkpoint.pt").exists():
        return TrainResult(**json.loads((root / "result.json").read_text()))
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2))

    def origins(start: str, end: str):
        for t in pd.date_range(start, end, freq=f"{config.origin_stride}D"):
            horizons = tuple(
                h for h in config.horizons if t + pd.Timedelta(days=h) <= pd.Timestamp(end)
            )
            if horizons:
                yield t, horizons

    prepared = []
    for t, horizons in origins(config.train_start, config.train_end):
        batch = data.build(t, horizons, config.shared_days)
        prepared.append((batch, base_forecast(data, batch, forecaster)))
    if not prepared:
        raise ValueError("No training origins with resolved labels")
    model_config = TwoStageConfig(hidden_dim=config.hidden_dim)
    model = TwoStagePaceNet(model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    result = TrainResult(
        NAME,
        VERSION,
        metadata,
        n_train_slots=sum(len(b.rows) for b, _ in prepared),
        run_dir=str(root),
    )
    rng = np.random.default_rng(config.seed)
    for _ in range(config.n_epochs):
        model.train()
        total = 0.0
        for i in rng.permutation(len(prepared)):
            batch, base = prepared[i]
            optimizer.zero_grad()
            prediction = model(batch, torch.from_numpy(base))
            target = torch.tensor(
                batch.rows.target.to_numpy(), dtype=torch.float32, device=config.device
            )
            cap = torch.tensor(
                batch.rows.capacity.to_numpy(), dtype=torch.float32, device=config.device
            )
            otb = torch.tensor(
                batch.rows.current_otb.to_numpy(), dtype=torch.float32, device=config.device
            )
            penalty = retention_penalty(prediction, otb, cap, config.min_retention).mean()
            loss = torch.nn.functional.mse_loss(prediction / cap, target / cap)
            loss = loss + config.retention_weight * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch.rows)
        result.train_loss_by_epoch.append(total / result.n_train_slots)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "metadata": metadata,
            },
            root / "checkpoint.pt",
        )
    tables = []
    for t, horizons in origins(config.test_start, config.test_end):
        forecast = predict(data, model, forecaster, str(t.date()), horizons, config.shared_days)
        forecast = forecast.merge(
            data.history.rename(columns={"rooms_sold": "target"}),
            on=["property_id", "stay_date"],
            how="left",
            validate="many_to_one",
        )
        tables.append(forecast)
    if not tables:
        raise ValueError("No test origins with resolved labels")
    predictions = pd.concat(tables, ignore_index=True)
    if predictions.target.isna().any():
        raise ValueError("Missing evaluation labels")
    predictions.to_parquet(root / "predictions.parquet", index=False)
    result.n_test_slots = len(predictions)
    result.test_metrics = _metrics(predictions, "prediction")
    result.baseline_metrics = _metrics(predictions, "base")
    result.otb_metrics = _metrics(predictions, "current_otb")
    result.save(root)
    return result
