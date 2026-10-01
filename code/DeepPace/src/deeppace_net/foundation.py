"""Frozen Chronos-2 forecasts from observed final rooms sold and calendar covariates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_sim.calendars import is_holiday


@dataclass(frozen=True)
class FoundationConfig:
    model_id: str = "amazon/chronos-2"
    revision: str = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    device: str = "cpu"
    context_days: int = 730
    min_history: int = 180
    cache_dir: str = "_runs/chronos_cache"


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Only deterministic, known-in-advance covariates; no simulator latent calendars."""
    return pd.DataFrame(
        {
            "timestamp": dates,
            "dow_sin": np.sin(2 * np.pi * dates.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * dates.dayofweek / 7),
            "year_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
            "year_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
            "holiday": [float(is_holiday(d.date())) for d in dates],
        }
    )


class ChronosForecaster:
    """Load lazily, freeze weights, and cache each actual input/forecast pair.

    Pass an immutable model revision for reproducible production experiments.
    A test pipeline may be injected without downloading weights.
    """

    def __init__(self, config: FoundationConfig | None = None, pipeline=None):
        self.config = config or FoundationConfig()
        self.pipeline = pipeline

    @property
    def identity(self) -> dict:
        return {
            "adapter_version": 2,
            "chronos_version": version("chronos-forecasting"),
            **asdict(self.config),
        }

    def predict(
        self, history: pd.DataFrame, as_of: pd.Timestamp, property_id: int, horizon: int
    ) -> np.ndarray:
        cfg = self.config
        if horizon < 1:
            raise ValueError("horizon must be positive")
        dates = pd.date_range(end=as_of, periods=cfg.context_days)
        values = history.set_index("stay_date")["rooms_sold"].reindex(dates)
        values = values.loc[values.first_valid_index() :] if values.notna().any() else values
        if values.notna().sum() < cfg.min_history or pd.isna(values.iloc[-1]):
            raise ValueError(f"property {property_id}: insufficient history through {as_of}")
        context = calendar(pd.DatetimeIndex(values.index))
        context["target"] = values.to_numpy()
        context["item_id"] = str(property_id)
        future = calendar(pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon))
        future["item_id"] = str(property_id)
        return self.predict_frames(context, future)

    def predict_frames(self, context: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        """Forecast from explicit, single-property as-of-safe frames.

        The caller owns covariate availability. This shared path keeps model revision,
        freezing, quantile choice and caching identical for history-only and pace cases.
        """
        cfg = self.config
        horizon = len(future)
        if horizon < 1 or context.item_id.nunique() != 1 or future.item_id.nunique() != 1:
            raise ValueError("Expected one property and a nonempty future frame")
        if context.item_id.iloc[0] != future.item_id.iloc[0]:
            raise ValueError("History and future property IDs differ")
        expected = pd.date_range(context.timestamp.max() + pd.Timedelta(days=1), periods=horizon)
        if not pd.DatetimeIndex(future.timestamp).equals(expected):
            raise ValueError("Future timestamps must follow the history on a daily grid")
        if "target" in future:
            raise ValueError("Future target values must never be passed to Chronos")
        payload = json.dumps(
            {
                "config": self.identity,
                "context": json.loads(context.to_json(date_format="iso")),
                "future": json.loads(future.to_json(date_format="iso")),
            },
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode()).hexdigest()
        root = Path(cfg.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{key}.npz"
        self.last_cache_path = path
        if path.exists():
            self.last_cache_hit = True
            with np.load(path) as saved:
                return saved["prediction"].copy()
        self.last_cache_hit = False
        if self.pipeline is None:
            from chronos import Chronos2Pipeline

            self.pipeline = Chronos2Pipeline.from_pretrained(
                cfg.model_id,
                revision=cfg.revision,
                device_map=cfg.device,
            )
            self.pipeline.model.eval()
            self.pipeline.model.requires_grad_(False)
        result = self.pipeline.predict_df(
            context,
            future_df=future,
            prediction_length=horizon,
            context_length=cfg.context_days,
            quantile_levels=[0.1, 0.5, 0.9],
            freq="D",
            cross_learning=False,
        )
        result = result.sort_values("timestamp")
        prediction = result["0.5"].to_numpy(dtype=np.float32)
        if len(prediction) != horizon or not np.isfinite(prediction).all():
            raise ValueError("Chronos returned invalid forecasts")
        np.savez_compressed(path, prediction=prediction)
        (root / f"{key}.json").write_text(payload)
        return prediction
