"""Ground-truth recovery diagnostics — what only a simulator makes possible.

Design: docs/design.md §7.3-§7.4.

| check | question |
|---|---|
| `mixture_recovery` (the decisive test) | does the model's learned state correspond to the true `w_group/leisure/lastmin`? |
| `completion_curve_shape`               | is pace actually being modelled, or just the level? |
| `unconstrained_demand_on_censored`     | is censoring handled, or are peaks systematically under-forecast? |
| `promo_sensitivity`                    | are the diagonal (booking-date) effects detected at all? |
| `mood_drift`                           | is persistent demand drift tracked, or absorbed as residual noise? |

**The decisive test** correlates the model's learned state against the true `w_*` columns via
a held-out linear probe (fit on one split of rows, scored on another, so the reported number
cannot be inflated by memorization). A context-conditioned encoder should show substantial
correlation; a context-free one near-zero. **An accuracy win with zero latent correlation does
not support the 2D claim** — report it as a separate number, not folded into an accuracy table.

Every check that cannot run (no model supplied, no promo calendar, degenerate variance, too
few rows) is reported as skipped, with why, rather than silently omitted — matching
`deeppace_sim.validate.ValidationReport`'s convention.

Public surface:
    LatentCheck, LatentReport (dataclasses)
    recover_latents(predictions, ground_truth, promo_calendar=None,
                     model=None, data=None, probe_batches=None) -> LatentReport
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIXTURE_COLUMNS = ("w_group", "w_leisure", "w_lastmin")


@dataclass(frozen=True)
class LatentCheck:
    name: str
    statistic: float
    n: int
    detail: str = ""
    skipped: str = ""


@dataclass(frozen=True)
class LatentReport:
    checks: list[LatentCheck] = field(default_factory=list)

    def get(self, name: str) -> LatentCheck:
        return next(c for c in self.checks if c.name == name)

    @property
    def decisive(self) -> LatentCheck:
        return self.get("mixture_recovery")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(c) for c in self.checks])


def _with_stay_date(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["stay_date"] = pd.to_datetime(frame["stay_date"])
    return frame


def _probe_states(states: np.ndarray, weights: np.ndarray, seed: int) -> dict[str, float] | None:
    """Held-out linear-probe correlation, per mixture component. `None` if underdetermined."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        states, weights, test_size=0.3, random_state=seed
    )
    probe = Ridge(alpha=1.0).fit(x_train, y_train)
    predicted = probe.predict(x_test)
    corrs = {
        name: float(np.corrcoef(predicted[:, j], y_test[:, j])[0, 1])
        for j, name in enumerate(MIXTURE_COLUMNS)
        if y_test[:, j].std() > 0 and predicted[:, j].std() > 0
    }
    return corrs or None


def _mixture_recovery(
    model, batches, ground_truth: pd.DataFrame, seed: int = 20260826,
) -> LatentCheck:
    """The decisive test: held-out linear-probe correlation of the local GRU state against
    true `w_*`, **relative to an untrained model with the same architecture**.

    A bare probe on `model`'s state is not decisive by itself: a GRU's hidden state is a smooth,
    information-preserving-ish function of its input even at random initialization, so a
    separately-fit linear probe recovers a substantial correlation from an *untrained* model
    too — verified directly on v3: an untrained `TwoStagePaceNet` probes to within 0.02 of a
    fully trained one (0.594/0.105/0.500 vs 0.574/0.128/0.498 per component). Reporting the raw
    trained-model number would have called that "decisive" when it demonstrates nothing about
    what training taught the model. The statistic here is `trained_corr - untrained_corr`; a
    value near zero means training added no mixture-recovery signal beyond what the raw curve
    already hands a random encoder, which is exactly the "accuracy win, zero latent
    correlation" failure mode CLAUDE.md warns against.
    """
    import torch

    gt = _with_stay_date(ground_truth)[["property_id", "stay_date", *MIXTURE_COLUMNS]]
    gt_index = gt.set_index(["property_id", "stay_date"])

    def encode_all(m) -> tuple[np.ndarray, np.ndarray]:
        states, weights = [], []
        for batch in batches:
            with torch.no_grad():
                hidden = m.encode(batch).cpu().numpy()
            keys = list(zip(batch.rows.property_id, batch.rows.stay_date, strict=True))
            w = gt_index.reindex(keys)[list(MIXTURE_COLUMNS)].to_numpy()
            valid = np.isfinite(w).all(axis=1)
            if valid.any():
                states.append(hidden[valid])
                weights.append(w[valid])
        return (np.concatenate(states), np.concatenate(weights)) if states else (
            np.empty((0, 0)), np.empty((0, 0))
        )

    x, y = encode_all(model)
    if len(x) < 20:
        return LatentCheck(
            "mixture_recovery", float("nan"), len(x),
            skipped=f"only {len(x)} probe rows; need >=20 for a held-out split",
        )
    trained_corrs = _probe_states(x, y, seed)

    torch.manual_seed(seed)
    untrained = model.__class__(model.config)
    x0, y0 = encode_all(untrained)
    untrained_corrs = _probe_states(x0, y0, seed) if len(x0) >= 20 else None

    if trained_corrs is None or untrained_corrs is None:
        return LatentCheck(
            "mixture_recovery", float("nan"), len(x),
            skipped="degenerate variance in the held-out probe split (trained or untrained)",
        )
    trained_mean = float(np.mean(list(trained_corrs.values())))
    untrained_mean = float(np.mean(list(untrained_corrs.values())))
    detail = (
        f"trained probe corr={trained_mean:.3f} "
        + ", ".join(f"{k}={v:.3f}" for k, v in trained_corrs.items())
        + f"; untrained-model control={untrained_mean:.3f} "
        + ", ".join(f"{k}={v:.3f}" for k, v in untrained_corrs.items())
    )
    return LatentCheck("mixture_recovery", trained_mean - untrained_mean, len(x), detail)


def _completion_curve_shape(predictions: pd.DataFrame) -> LatentCheck:
    """Does the *shape* of the model's remaining-pickup prediction track the true completion
    curve across regimes (archetype x horizon), or only the overall level?"""
    df = predictions.copy()
    df["pred_remaining"] = (df["prediction"] - df["current_otb"]) / df["capacity"]
    df["true_remaining"] = (df["target"] - df["current_otb"]) / df["capacity"]
    cells = df.groupby(["archetype", "horizon"])[["pred_remaining", "true_remaining"]].mean()
    if len(cells) < 4 or cells["true_remaining"].std() == 0 or cells["pred_remaining"].std() == 0:
        return LatentCheck(
            "completion_curve_shape", float("nan"), len(cells),
            skipped="fewer than 4 (archetype, horizon) cells, or no variation to correlate",
        )
    corr = float(np.corrcoef(cells["pred_remaining"], cells["true_remaining"])[0, 1])
    detail = f"corr across {len(cells)} (archetype, horizon) cells of mean predicted vs. true remaining-pickup fraction"
    return LatentCheck("completion_curve_shape", corr, len(cells), detail)


def _unconstrained_demand_on_censored(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame,
) -> LatentCheck:
    """On dates the simulator actually censored, does the forecast show any sign of the true,
    higher demand — or does it just track the capped label it was trained against?"""
    gt = _with_stay_date(ground_truth)[["property_id", "stay_date", "unconstrained_demand", "turned_away"]]
    merged = predictions.merge(gt, on=["property_id", "stay_date"], how="inner")
    censored = merged[merged["turned_away"] > 0]
    if censored.empty:
        return LatentCheck(
            "unconstrained_demand_on_censored", float("nan"), 0,
            skipped="no censored stay dates in this prediction set",
        )
    bias_vs_unconstrained = float((censored["prediction"] - censored["unconstrained_demand"]).mean())
    bias_vs_target = float((censored["prediction"] - censored["target"]).mean())
    detail = (
        f"bias vs true unconstrained_demand={bias_vs_unconstrained:.2f} rooms "
        f"(bias vs the capped training target={bias_vs_target:.2f}); "
        "more negative vs. unconstrained than vs. target means peaks are under-forecast, "
        "not just correctly matching the censored label"
    )
    return LatentCheck("unconstrained_demand_on_censored", bias_vs_unconstrained, len(censored), detail)


def _promo_sensitivity(
    predictions: pd.DataFrame, promo_calendar: pd.DataFrame | None, window_days: int = 7,
) -> LatentCheck:
    """Compares the model's adjustment on as-of dates within `window_days` of a planted promo
    booking date against dates with no recent promo — the diagonal-effect detector."""
    if promo_calendar is None or promo_calendar.empty:
        return LatentCheck("promo_sensitivity", float("nan"), 0, skipped="no promo_calendar.parquet")
    promo_dates = set(pd.to_datetime(promo_calendar["booking_date"]).dt.normalize())
    as_of = pd.to_datetime(predictions["as_of"])
    recent_promo = np.zeros(len(predictions), dtype=bool)
    for offset in range(window_days):
        recent_promo |= (as_of - pd.Timedelta(days=offset)).isin(promo_dates)
    signal = (
        predictions["adjustment"] if "adjustment" in predictions
        else predictions["prediction"] - predictions["current_otb"]
    )
    treated, control = signal[recent_promo], signal[~recent_promo]
    if treated.empty or control.empty:
        return LatentCheck(
            "promo_sensitivity", float("nan"), len(predictions),
            skipped="no promo-touched rows, or no control rows, in this window",
        )
    diff = float(treated.mean() - control.mean())
    detail = (
        f"mean adjustment, as-of within {window_days}d of a promo booking date "
        f"(n={len(treated)}) minus control (n={len(control)}) = {diff:.3f} rooms"
    )
    return LatentCheck("promo_sensitivity", diff, len(treated), detail)


def _mood_drift(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> LatentCheck:
    """Correlates forecast error with the true AR(1) mood path — near zero means persistent
    demand drift is being tracked; a strong correlation means it's leaking into the residual."""
    gt = _with_stay_date(ground_truth)[["property_id", "stay_date", "mood"]]
    merged = predictions.merge(gt, on=["property_id", "stay_date"], how="inner")
    if merged.empty:
        return LatentCheck("mood_drift", float("nan"), 0, skipped="no ground_truth rows joined")
    error = merged["prediction"] - merged["target"]
    if error.std() == 0 or merged["mood"].std() == 0:
        return LatentCheck("mood_drift", float("nan"), len(merged), skipped="degenerate variance")
    corr = float(np.corrcoef(error, merged["mood"])[0, 1])
    detail = f"corr(prediction - target, true mood), n={len(merged)}"
    return LatentCheck("mood_drift", corr, len(merged), detail)


def recover_latents(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    promo_calendar: pd.DataFrame | None = None,
    model=None,
    batches=None,
    seed: int = 20260826,
) -> LatentReport:
    """Run every §7.3 check. `predictions` needs `property_id, stay_date, as_of, horizon,
    archetype, capacity, current_otb, target, prediction` (the schema `train.py`/`benchmark.py`
    already write). The decisive test additionally needs `model` (a `TwoStagePaceNet`, for
    `.encode()`) and `batches` (the `PaceBatch` list the predictions were computed from) — pass
    both, or it is reported as skipped rather than silently omitted.
    """
    if model is not None and batches:
        decisive = _mixture_recovery(model, batches, ground_truth, seed)
    else:
        decisive = LatentCheck(
            "mixture_recovery", float("nan"), 0,
            skipped="no model/batches supplied — pass model= and batches= to run the decisive test",
        )
    return LatentReport([
        decisive,
        _completion_curve_shape(predictions),
        _unconstrained_demand_on_censored(predictions, ground_truth),
        _promo_sensitivity(predictions, promo_calendar),
        _mood_drift(predictions, ground_truth),
    ])
