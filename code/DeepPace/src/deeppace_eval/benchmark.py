"""Four-case Chronos/pace benchmark with validation-only selection and paired test rows.

Usage:
    uv run --extra net python -m scripts.evaluate_net --data _data/v3
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_eval.comparison import paired_comparisons, statistics, summarize, validate_alignment
from deeppace_eval.covariates import COMPACT_GRID, build_covariates
from deeppace_eval.profiling import Measure
from deeppace_net.foundation import ChronosForecaster, FoundationConfig
from deeppace_net.pace_data import PaceBatch, PaceData
from deeppace_sim.dataset import LoadedDataset

VERSION = 1
CASES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class EvaluationConfig:
    train_start: str = "2023-07-01"
    train_end: str = "2024-06-30"  # label cutoff
    validation_start: str = "2024-07-01"
    validation_end: str = "2024-12-31"  # label cutoff
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"  # label cutoff
    origin_stride: int = 7
    train_stride: int = 30
    horizons: tuple[int, ...] = (1, 7, 10, 30, 90, 180)
    compact_grids: tuple[tuple[int, ...], ...] = (
        COMPACT_GRID, (1, 7, 14, 30, 60, 90, 180, 365),
    )
    hidden_dims: tuple[int, ...] = (16,)
    epochs: tuple[int, ...] = (10, 20)
    lr: float = 1e-3
    min_retention: float = 0.5  # see deeppace_net.models.two_stage.retention_penalty
    retention_weight: float = 0.2
    shared_days: int = 90
    include_shared: bool = True
    property_ids: tuple[int, ...] | None = None
    bootstrap_draws: int = 1000
    block_days: int = 180
    seed: int = 20260826
    device: str = "cpu"
    out_dir: str = "_runs/four_case_eval"
    foundation: FoundationConfig = field(default_factory=FoundationConfig)

    def validate(self, max_dta: int) -> None:
        ts, te, vs, ve, ss, se = map(
            pd.Timestamp,
            (
                self.train_start,
                self.train_end,
                self.validation_start,
                self.validation_end,
                self.test_start,
                self.test_end,
            ),
        )
        if not ts < te < vs < ve < ss < se:
            raise ValueError(
                "Require train start < label cutoff < validation start "
                "< cutoff < test start < cutoff"
            )
        if (
            min(
                self.origin_stride,
                self.train_stride,
                self.bootstrap_draws,
                self.block_days,
                self.shared_days,
            )
            < 1
        ):
            raise ValueError("Strides, bootstrap settings and shared_days must be positive")
        if (
            not self.horizons
            or len(set(self.horizons)) != len(self.horizons)
            or min(self.horizons) < 1
            or max(self.horizons) > max_dta
        ):
            raise ValueError("Horizons must be unique and inside the daily DTA grid")
        if not self.hidden_dims or not self.epochs or min((*self.hidden_dims, *self.epochs)) < 1:
            raise ValueError("At least one positive hidden dimension and epoch count is required")
        if not self.compact_grids or any(
            not g or min(g) < 1 or max(g) > max_dta for g in self.compact_grids
        ):
            raise ValueError("Compact grids must contain valid positive DTA points")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")


@dataclass(frozen=True)
class EvaluationResult:
    run_dir: str
    selected_grid: tuple[int, ...]
    selected_models: dict
    cases: tuple[str, ...]
    n_test_rows_per_case: int


def origin_schedule(start: str, end: str, stride: int, horizons: tuple[int, ...]):
    for origin in pd.date_range(start, end, freq=f"{stride}D"):
        available = tuple(h for h in horizons if origin + pd.Timedelta(days=h) <= pd.Timestamp(end))
        if available:
            yield origin, available


def without_shared(batch: PaceBatch) -> PaceBatch:
    """Constant shared input: D uses no other property's or arrival date's activity."""
    return replace(batch, shared=np.zeros_like(batch.shared))


class Benchmark:
    def __init__(
        self,
        ds: LoadedDataset,
        config: EvaluationConfig,
        forecaster: ChronosForecaster | None = None,
    ):
        self.config = config
        self.data = PaceData(ds, list(config.property_ids) if config.property_ids else None)
        config.validate(self.data.k)
        if self.data.properties.empty:
            raise ValueError("No properties selected")
        self.forecaster = forecaster or ChronosForecaster(config.foundation)
        if self.forecaster.config != config.foundation:
            raise ValueError("Foundation config must match the evaluation configuration")
        digest = hashlib.sha256()
        for frame in (self.data.history, self.data.properties, self.data.surface):
            digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        source = hashlib.sha256()
        package_root = Path(__file__).resolve().parents[1]
        for relative in ("deeppace_eval/benchmark.py", "deeppace_eval/covariates.py",
                         "deeppace_eval/comparison.py", "deeppace_eval/profiling.py",
                         "deeppace_net/foundation.py", "deeppace_net/pace_data.py",
                         "deeppace_net/models/two_stage.py"):
            source.update((package_root / relative).read_bytes())
        self.manifest = {
            "evaluation_version": VERSION,
            "config": asdict(config),
            "dataset_version": ds.dataset_version,
            "generator_hash": ds.generator_hash,
            "data_hash": digest.hexdigest(),
            "foundation": self.forecaster.identity,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_hash": source.hexdigest(),
            "packages": {name: version(name) for name in
                         ("torch", "numpy", "pandas", "chronos-forecasting")},
        }
        key = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()[:16]
        self.root = Path(config.out_dir) / key
        self.root.mkdir(parents=True, exist_ok=True)
        self.costs: list[dict] = []
        self.history_counts: dict[tuple[int, pd.Timestamp], tuple[int, int]] = {}

    def direct(
        self,
        t: pd.Timestamp,
        horizons: tuple[int, ...],
        case: str,
        phase: str,
        grid: tuple[int, ...] = COMPACT_GRID,
    ) -> np.ndarray:
        values = []
        for pid in self.data.properties.property_id:
            for h in horizons:
                with Measure() as usage:
                    frames = build_covariates(
                        self.data, int(pid), t, h, self.config.foundation, case, grid
                    )
                    loaded_before = self.forecaster.pipeline is not None
                    forecast = self.forecaster.predict_frames(frames.history, frames.future)
                self.history_counts[(int(pid), t)] = (
                    frames.history_days,
                    frames.observed_history_days,
                )
                cache_path = self.forecaster.last_cache_path
                cache_bytes = sum(p.stat().st_size for p in
                                  (cache_path, cache_path.with_suffix(".json")) if p.exists())
                self.costs.append(
                    {
                        "phase": phase,
                        "case": case,
                        "operation": "foundation",
                        "property_id": int(pid),
                        "as_of": str(t.date()),
                        "horizon": h,
                        "cache_hit": self.forecaster.last_cache_hit,
                        "model_loaded_before": loaded_before,
                        "cache_key": cache_path.stem,
                        "cache_bytes": cache_bytes,
                        **asdict(usage),
                    }
                )
                values.append(forecast[-1])
        return np.asarray(values, dtype=np.float32)

    def prepare(self, start: str, end: str, stride: int, phase: str):
        batches = []
        for t, horizons in origin_schedule(start, end, stride, self.config.horizons):
            batch = self.data.build(t, horizons, self.config.shared_days, labeled=True)
            if not np.isfinite(batch.rows.target).all():
                raise ValueError("Missing labels in scheduled examples")
            base = self.direct(t, horizons, "A", phase)
            batches.append((batch, base))
        if not batches:
            raise ValueError(f"No scheduled examples in {phase}")
        return batches

    def table(self, batch: PaceBatch, raw: np.ndarray, case: str) -> pd.DataFrame:
        result = batch.rows.copy()
        result["case"] = case
        result["raw_prediction"] = raw
        result["prediction"] = np.clip(raw, 0, result.capacity)
        result["occupancy_band"] = pd.cut(
            result.target / result.capacity,
            [-np.inf, 0.4, 0.8, 0.95, np.inf],
            labels=["low_<40%", "40-80%", "80-95%", "near_capacity_>=95%"],
            right=False,
        ).astype(str)
        result["history_days"] = [
            self.history_counts[(int(p), t)][0]
            for p, t in zip(result.property_id, result.as_of, strict=True)
        ]
        result["observed_history_days"] = [
            self.history_counts[(int(p), t)][1]
            for p, t in zip(result.property_id, result.as_of, strict=True)
        ]
        return result

    def fit(self, batches, hidden: int, epochs: int, shared: bool):
        import torch

        from deeppace_net.models.two_stage import (
            TwoStageConfig,
            TwoStagePaceNet,
            retention_penalty,
        )

        cfg = self.config
        tag = f"{'shared' if shared else 'local'}-h{hidden}-e{epochs}"
        path = self.root / f"{tag}.pt"
        model = TwoStagePaceNet(TwoStageConfig(hidden_dim=hidden)).to(cfg.device)
        if path.exists():
            saved = torch.load(path, map_location=cfg.device, weights_only=True)
            model.load_state_dict(saved["state_dict"])
            return model, tag
        torch.manual_seed(cfg.seed)
        model = TwoStagePaceNet(TwoStageConfig(hidden_dim=hidden)).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        rng = np.random.default_rng(cfg.seed)
        loss_log = []
        with Measure() as usage:
            for _ in range(epochs):
                model.train()
                total, n = 0.0, 0
                for i in rng.permutation(len(batches)):
                    original, base = batches[i]
                    batch = original if shared else without_shared(original)
                    opt.zero_grad()
                    pred = model(batch, torch.from_numpy(base))
                    target = torch.tensor(
                        batch.rows.target.to_numpy(), dtype=torch.float32, device=cfg.device
                    )
                    cap = torch.tensor(
                        batch.rows.capacity.to_numpy(), dtype=torch.float32, device=cfg.device
                    )
                    otb = torch.tensor(
                        batch.rows.current_otb.to_numpy(), dtype=torch.float32, device=cfg.device
                    )
                    penalty = retention_penalty(pred, otb, cap, cfg.min_retention).mean()
                    loss = torch.nn.functional.mse_loss(pred / cap, target / cap)
                    loss = loss + cfg.retention_weight * penalty
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total += loss.item() * len(base)
                    n += len(base)
                loss_log.append(total / n)
        self.costs.append(
            {
                "phase": "train",
                "case": "D_shared" if shared else "D",
                "operation": "adjustment_training",
                "candidate": tag,
                **asdict(usage),
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": asdict(model.config),
                "shared": shared,
                "loss": loss_log,
                "metadata": self.manifest,
            },
            path,
        )
        return model, tag

    def adjusted(self, batches, model, shared: bool, phase: str) -> pd.DataFrame:
        import torch

        model.eval()
        case = "D_shared" if shared else "D"
        tables = []
        for original, base in batches:
            batch = original if shared else without_shared(original)
            with Measure() as usage, torch.no_grad():
                raw = model(batch, torch.from_numpy(base)).cpu().numpy()
            self.costs.append(
                {"phase": phase, "case": case, "operation": "adjustment_inference", **asdict(usage)}
            )
            tables.append(self.table(batch, raw, case))
        return pd.concat(tables, ignore_index=True)

    def direct_tables(self, batches, case: str, phase: str, grid=COMPACT_GRID):
        tables = []
        for batch, base in batches:
            t = pd.Timestamp(batch.rows.as_of.iloc[0])
            horizons = tuple(int(h) for h in batch.rows.horizon.unique())
            raw = base if case == "A" else self.direct(t, horizons, case, phase, grid)
            tables.append(self.table(batch, raw, case))
        return pd.concat(tables, ignore_index=True)

    def run(self) -> EvaluationResult:
        path = self.root / "result.json"
        if path.exists():
            obj = json.loads(path.read_text())
            return EvaluationResult(
                **{
                    **obj,
                    "selected_grid": tuple(obj["selected_grid"]),
                    "cases": tuple(obj["cases"]),
                }
            )
        cfg = self.config
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        training = self.prepare(cfg.train_start, cfg.train_end, cfg.train_stride, "train")
        validation = self.prepare(
            cfg.validation_start, cfg.validation_end, cfg.origin_stride, "validation"
        )
        selection, validation_tables = [], []
        grid_scores = []
        for i, grid in enumerate(cfg.compact_grids):
            table = self.direct_tables(validation, "C", "validation", grid)
            score = statistics(table.target, table.prediction)["wmape"]
            if not np.isfinite(score):
                raise ValueError("Validation WMAPE is undefined; cannot select a model")
            grid_scores.append(score)
            selection.append(
                {"case": "C", "candidate": f"grid_{i}", "grid": list(grid), "wmape": score}
            )
            table["candidate"] = f"grid_{i}"
            validation_tables.append(table)
        selected_grid = cfg.compact_grids[int(np.argmin(grid_scores))]
        chosen_models, model_tags = {}, {}
        for shared in (False, True) if cfg.include_shared else (False,):
            case = "D_shared" if shared else "D"
            best = np.inf
            for hidden in cfg.hidden_dims:
                for epochs in cfg.epochs:
                    model, tag = self.fit(training, hidden, epochs, shared)
                    table = self.adjusted(validation, model, shared, "validation")
                    value = statistics(table.target, table.prediction)["wmape"]
                    if not np.isfinite(value):
                        raise ValueError("Validation WMAPE is undefined")
                    selection.append({"case": case, "candidate": tag, "wmape": value})
                    table["candidate"] = tag
                    validation_tables.append(table)
                    if value < best:
                        best, chosen_models[case], model_tags[case] = value, model, tag
        # Choices are persisted before any test forecasts/labels are scored.
        (self.root / "selection.json").write_text(
            json.dumps(
                {
                    "candidates": selection,
                    "selected_grid": selected_grid,
                    "selected_models": model_tags,
                },
                indent=2,
            )
        )
        for case in ("A", "B"):
            validation_tables.append(self.direct_tables(validation, case, "validation"))
        pd.concat(validation_tables, ignore_index=True).to_parquet(
            self.root / "validation.parquet", index=False
        )
        test = self.prepare(cfg.test_start, cfg.test_end, cfg.origin_stride, "test")
        tables = [self.direct_tables(test, case, "test", selected_grid) for case in ("A", "B", "C")]
        for case, model in chosen_models.items():
            tables.append(self.adjusted(test, model, case == "D_shared", "test"))
        predictions = pd.concat(tables, ignore_index=True)
        cases = CASES + (("D_shared",) if cfg.include_shared else ())
        validate_alignment(predictions, cases)
        predictions.to_parquet(self.root / "predictions.parquet", index=False)
        metrics = summarize(predictions)
        metrics.to_csv(self.root / "metrics.csv", index=False)
        pairs = paired_comparisons(
            predictions, cases, cfg.bootstrap_draws, cfg.block_days, cfg.seed
        )
        pairs.to_csv(self.root / "comparisons.csv", index=False)
        costs = pd.DataFrame(self.costs)
        costs.to_csv(self.root / "runtime.csv", index=False)
        cost_summary = []
        for case in cases:
            own = costs[costs.case == case]
            relevant = costs[costs.case.isin([case, "A"])] if case.startswith("D") else own
            inference = relevant[relevant.phase == "test"]
            cache = relevant.dropna(subset=["cache_key"]).drop_duplicates("cache_key")
            checkpoint_prefix = "shared" if case == "D_shared" else "local"
            checkpoint_bytes = (sum(p.stat().st_size for p in self.root.glob(f"{checkpoint_prefix}-*.pt"))
                                if case.startswith("D") else 0)
            cost_summary.append({"case": case, "test_inference_seconds": inference.seconds.sum(),
                "test_cache_hit_calls": int(inference.cache_hit.fillna(False).astype(bool).sum()),
                "test_calls": len(inference), "sampled_peak_cpu_rss_bytes": relevant.rss_peak_bytes.max(),
                "adjustment_training_seconds": own[own.operation == "adjustment_training"].seconds.sum(),
                "referenced_cache_bytes": cache.cache_bytes.sum(), "candidate_checkpoint_bytes": checkpoint_bytes})
        pd.DataFrame(cost_summary).to_csv(self.root / "cost_summary.csv", index=False)
        coverage = []
        for phase, batches in (("train", training), ("validation", validation), ("test", test)):
            frame = pd.concat([b.rows for b, _ in batches], ignore_index=True)
            for h, part in frame.groupby("horizon"):
                coverage.append({"phase": phase, "horizon": h, "n": len(part),
                                 "n_origins": part.as_of.nunique(), "first_origin": part.as_of.min(),
                                 "last_origin": part.as_of.max(), "last_label": part.stay_date.max()})
        pd.DataFrame(coverage).to_csv(self.root / "coverage.csv", index=False)
        self.report(metrics, pairs, predictions)
        result = EvaluationResult(
            str(self.root),
            selected_grid,
            model_tags,
            cases,
            len(predictions[predictions.case == "A"]),
        )
        path.write_text(json.dumps(asdict(result), indent=2))
        return result

    def report(self, metrics, pairs, predictions):
        lines = [
            "# Four-case final rooms-sold evaluation",
            "",
            "A: history/calendar only; B: full DTA covariates; C: validation-selected compact",
            "DTA covariates plus horizon-specific OTB; D: trained local pace adjustment.",
            "D_shared, when included, additionally sees portfolio booking activity and is an",
            "information ablation, not an equal-information architecture comparison.",
            "",
            "| Case | WMAPE | MAE | Bias | MAPE | N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in metrics[metrics.dimension == "all"].itertuples():
            lines.append(
                f"| {row.case} | {row.wmape:.4f} | {row.mae:.3f} | "
                f"{row.bias:.3f} | {row.mape:.4f} | {row.n} |"
            )
        lines += [
            "",
            "Paired differences in comparisons.csv are candidate minus reference; negative",
            "means lower error. 95% intervals use circular moving blocks of forecast origins,",
            "keeping properties/horizons paired. Intervals are conditional on this portfolio.",
            f"Block span: {self.config.block_days} days. CI unavailable for "
            f"{int((pairs.ci_status != 'ok').sum())} comparisons with fewer than two blocks.",
            "",
            "Exact-horizon, archetype, occupancy-band and property scores are in metrics.csv.",
            "Zero targets are excluded only from MAPE; all-zero groups have undefined WMAPE.",
            "Ties are not counted as wins. Occupancy bands use final outcomes for reporting only.",
            "",
            "## Coverage",
            "",
            "| Horizon | Test origins | Rows per case |",
            "|---|---:|---:|",
        ]
        for h, part in predictions[predictions.case == "A"].groupby("horizon"):
            lines.append(f"| {h} | {part.as_of.nunique()} | {len(part)} |")
        lines += [
            "",
            f"History length range: {predictions.history_days.min()}"
            f"–{predictions.history_days.max()} days.",
            "Training labels resolve before validation origins; validation labels resolve before",
            "test origins. Test observations may enter later context, but weights remain fixed.",
            "coverage.csv records training, validation and test coverage separately.",
            "",
            "## Cost",
            "",
            "runtime.csv separates training, validation and test calls, cache hits and model",
            "load state. D's total inference cost includes A's forecast plus its adjustment.",
            "RSS is sampled process resident memory (10 ms), includes native CPU allocations and",
            "resident weights, and is not accelerator peak memory. Cold/warm/cache-hit timings",
            "must not be pooled as an intrinsic model-speed ranking.",
            "cost_summary.csv records per-case time, sampled CPU peak RSS, referenced cache bytes",
            "and candidate checkpoint bytes. D includes A's forecasting cost. Cache space is",
            "shared, so per-case cache sizes must not be added. Costs describe this execution,",
            "not a standardized cold-start speed benchmark. No adjusted intervals are claimed.",
        ]
        (self.root / "report.md").write_text("\n".join(lines) + "\n")


def evaluate(
    ds: LoadedDataset,
    config: EvaluationConfig | None = None,
    forecaster: ChronosForecaster | None = None,
) -> EvaluationResult:
    return Benchmark(ds, config or EvaluationConfig(), forecaster).run()
