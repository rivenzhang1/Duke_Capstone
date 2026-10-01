"""Aligned metrics and paired moving-block bootstrap across forecast origins."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

KEYS = ["property_id", "as_of", "horizon"]


def validate_alignment(predictions: pd.DataFrame, cases: tuple[str, ...]) -> None:
    """Fail rather than silently score different rows or duplicate forecasts."""
    reference = None
    for case in cases:
        rows = predictions[predictions.case == case]
        if rows.empty or rows.duplicated(KEYS).any():
            raise ValueError(f"Missing or duplicate forecast rows for {case}")
        indexed = rows.set_index(KEYS).sort_index()
        if not np.isfinite(indexed[["target", "prediction", "raw_prediction"]]).all().all():
            raise ValueError(f"Non-finite labels/predictions for {case}")
        truth = indexed[["stay_date", "target", "capacity"]]
        if reference is not None and not truth.equals(reference):
            raise ValueError(f"Forecast row/label alignment differs for {case}")
        reference = truth


def statistics(target: np.ndarray, pred: np.ndarray) -> dict:
    target, pred = np.asarray(target), np.asarray(pred)
    error = pred - target
    denominator = np.abs(target).sum()
    nz = target != 0
    return {
        "n": len(target),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "wmape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "mape": float(np.abs(error[nz] / target[nz]).mean()) if nz.any() else np.nan,
        "mape_n": int(nz.sum()),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, frame in predictions.groupby("case"):
        groups = [("all", "all", frame)]
        for dimension in ("horizon", "archetype", "occupancy_band", "property_id"):
            groups.extend((dimension, str(value), part) for value, part in frame.groupby(dimension))
        for dimension, value, part in groups:
            rows.append(
                {
                    "case": case,
                    "dimension": dimension,
                    "segment": value,
                    "n_origins": part.as_of.nunique(),
                    **statistics(part.target, part.prediction),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    predictions: pd.DataFrame,
    cases: tuple[str, ...],
    draws: int = 1000,
    block_days: int = 180,
    seed: int = 20260826,
) -> pd.DataFrame:
    """Resample contiguous origin blocks, keeping all properties/horizons paired.

    CI is conditional on this portfolio. Fewer than two blocks yields no CI, not a
    misleading zero-width interval. Negative delta means the first named model wins.
    """
    validate_alignment(predictions, cases)
    if draws < 1 or block_days < 1:
        raise ValueError("Bootstrap draws and block_days must be positive")
    rng = np.random.default_rng(seed)
    output = []
    for reference, candidate in combinations(cases, 2):
        a = predictions[predictions.case == reference].set_index(KEYS).sort_index()
        b = predictions[predictions.case == candidate].set_index(KEYS).sort_index()
        paired = a.reset_index()[KEYS + ["target"]].copy()
        paired["ea"] = np.abs(a.prediction.to_numpy() - a.target.to_numpy())
        paired["eb"] = np.abs(b.prediction.to_numpy() - b.target.to_numpy())
        groups = [("all", paired)] + [(f"horizon_{h}", p) for h, p in paired.groupby("horizon")]
        for segment, part in groups:
            daily = part.groupby("as_of", sort=True).agg(
                ea=("ea", "sum"),
                eb=("eb", "sum"),
                denom=("target", lambda x: x.abs().sum()),
                n=("target", "size"),
            )
            dates = pd.DatetimeIndex(daily.index)
            n = len(daily)
            step = (
                float(np.median(np.diff(dates).astype("timedelta64[D]").astype(int)))
                if n > 1
                else 1
            )
            length = max(1, int(np.ceil(block_days / step)))
            eligible = n >= 2 * length
            delta = daily.eb.sum() - daily.ea.sum()
            denominator = daily.denom.sum()
            intervals = {
                "wmape_low": np.nan,
                "wmape_high": np.nan,
                "mae_low": np.nan,
                "mae_high": np.nan,
            }
            if eligible:
                samples = []
                values = daily[["ea", "eb", "denom", "n"]].to_numpy()
                for _ in range(draws):
                    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
                    indices = ((starts[:, None] + np.arange(length)) % n).ravel()[:n]
                    sums = values[indices].sum(axis=0)
                    gap = sums[1] - sums[0]
                    samples.append([gap / sums[2] if sums[2] else np.nan, gap / sums[3]])
                samples = np.asarray(samples)
                for j, metric in enumerate(("wmape", "mae")):
                    finite = samples[:, j][np.isfinite(samples[:, j])]
                    if len(finite):
                        intervals[f"{metric}_low"], intervals[f"{metric}_high"] = np.quantile(
                            finite, [0.025, 0.975]
                        )
            prop = part.groupby("property_id")[["ea", "eb"]].mean()
            hs = part.groupby("horizon")[["ea", "eb"]].mean()
            output.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "segment": segment,
                    "delta_wmape": delta / denominator if denominator else np.nan,
                    "delta_mae": delta / len(part),
                    "n_origins": n,
                    "block_origins": length,
                    "ci_status": "ok" if eligible else "insufficient_blocks",
                    "property_win_rate": float((prop.eb < prop.ea).mean()),
                    "horizon_win_rate": float((hs.eb < hs.ea).mean()),
                    "row_win_rate": float((part.eb < part.ea).mean()),
                    "row_tie_rate": float((part.eb == part.ea).mean()),
                    **intervals,
                }
            )
    return pd.DataFrame(output)
