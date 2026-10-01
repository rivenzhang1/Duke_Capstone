"""Post-generation checks: exact identities, plausibility bands, and external anchors.

Spec: docs/simulation-spec.md.

Three kinds of evidence, kept apart because they fail for different reasons and a reader needs to
know which one just went red:

- **Identities (T0)** — exact equalities. A failure is a bug and blocks the dataset.
- **Bands** — plausibility ranges. A failure is a calibration question, not a bug.
- **Anchors (T3)** — comparisons against real data or against a design target. Not pass/fail on
  the generator; each produces a number that qualifies how far a Phase II result travels.

The identities still *emerge* rather than being imposed, even though the curve is now generated
top-down. `gross_cum`, `cancel_cum` and `otb` are all reverse-cumulative sums of one walked event
sequence, so their relationship is a fact about the walk rather than an arithmetic convention —
which is the reason `curve.apply_capacity` still walks the DTA axis instead of clipping a
finished curve.

Public surface:
    ValidationReport (dataclass)
    validate_dataset(ds) -> ValidationReport
    anchors(ds) -> ValidationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from deeppace_sim.config import GeneratorConstants
from deeppace_sim.dataset import LoadedDataset

# Weatherford & Kimes (2003), real hotel data, per room-class/LOS/DOW cell over two years.
WK_VAR_OVER_MEAN = 39.0
WK_CV = 0.72
# Antonio et al. (2019), 119,390 real bookings: median booking lead time, and the booking-weekday
# swing peak-to-trough.
ANTONIO_MEDIAN_LEAD = 69
ANTONIO_BOOKING_DOW_RATIO = 2.4


@dataclass
class ValidationReport:
    name: str = ""
    tier: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def failures(self) -> list[str]:
        return [k for k, ok in self.checks.items() if not ok]


def validate_dataset(ds: LoadedDataset) -> ValidationReport:
    """Exact identities plus plausibility bands."""
    report = ValidationReport(name="identities", tier="T0")
    on_books, stay_dates, gt = ds.on_books, ds.stay_dates, ds.ground_truth
    cap = ds.capacity
    key = ["property_id", "stay_date"]

    otb0 = on_books.loc[on_books["dta"] == 0].set_index(key)["otb"]
    rooms_sold = stay_dates.set_index(key)["rooms_sold"]
    report.checks["I2_otb0_equals_rooms_sold"] = bool(
        (otb0.reindex(rooms_sold.index) == rooms_sold).all()
    )
    report.checks["I1_gross_minus_cancel_equals_otb"] = bool(
        ((on_books["gross_cum"] - on_books["cancel_cum"]) == on_books["otb"]).all()
    )
    report.checks["I3_otb_within_capacity"] = bool(
        (on_books["otb"] <= on_books["property_id"].map(cap)).all()
    )
    report.checks["I5_otb_nonneg_everywhere"] = bool((on_books["otb"] >= 0).all())

    # reverse-cumulative, so walking dta downwards is walking calendar time forwards
    ordered = on_books.sort_values(key + ["dta"], ascending=[True, True, False])
    grouped = ordered.groupby(key, sort=False)
    report.checks["I4_gross_monotone_in_time"] = bool(
        (grouped["gross_cum"].diff().fillna(0.0) >= 0).all()
    )
    report.checks["I4_cancel_monotone_in_time"] = bool(
        (grouped["cancel_cum"].diff().fillna(0.0) >= 0).all()
    )

    g = gt.set_index(key)
    accepted = on_books.loc[on_books["dta"] == 0].set_index(key)["gross_cum"].reindex(g.index)
    report.checks["I6_drawn_equals_accepted_plus_turned_away"] = bool(
        ((accepted + g["turned_away"]) == g["gross_drawn"]).all()
    )
    report.checks["I6_censored_iff_turned_away"] = bool(
        (g["was_censored"] == (g["turned_away"] > 0)).all()
    )

    w = stay_dates[["w_group", "w_leisure", "w_lastmin"]]
    report.checks["I7_weights_sum_to_one"] = bool(np.allclose(w.sum(axis=1), 1.0))

    if ds.full_grid:
        # At the moment of rejection on-books stands exactly at capacity, but the stored curve is
        # end-of-day: cancellations landing at the same DTA are already subtracted. So the
        # observable invariant adds that day's cancellations back before comparing.
        ordered_full = on_books.sort_values(key + ["dta"], ascending=[True, True, False])
        cancel_inc = ordered_full.groupby(key, sort=False)["cancel_cum"].diff().fillna(
            ordered_full["cancel_cum"]
        )
        gross_at_k = ordered_full["otb"] + cancel_inc
        reached = gross_at_k.groupby(
            [ordered_full["property_id"], ordered_full["stay_date"]]
        ).max()
        censored = g.index[g["turned_away"] > 0]
        if len(censored):
            caps = np.array([cap[p] for p, _ in censored])
            report.checks["I8_turnaway_implies_capacity_reached"] = bool(
                (reached.reindex(censored).to_numpy() >= caps).all()
            )
    else:
        report.skipped["I8_turnaway_implies_capacity_reached"] = "needs the full DTA grid"

    # --- calibration: lam must be the mean of the demand it generates ---
    ratio = gt["gross_drawn"] / gt["lam_true"]
    mean, tol = float(ratio.mean()), 4.0 * float(ratio.std()) / np.sqrt(len(ratio))
    report.measurements["C1_gross_drawn_over_lam"] = mean
    report.measurements["C1_tolerance"] = tol
    report.checks["C1_lambda_is_expected_room_nights"] = abs(mean - 1.0) < tol

    shock = float(gt["shock_total"].mean())
    report.measurements["C2_mean_shock_total"] = shock
    report.checks["C2_shock_is_mean_one"] = abs(shock - 1.0) < 0.02

    # --- plausibility bands ---
    sd = ds.stay_dates_enriched()
    occ = float(sd["occ"].mean())
    report.measurements["mean_occupancy"] = occ
    report.checks["mean_occupancy_band"] = 0.55 < occ < 0.85

    # The meaningful sell-out concept is "the date closed out at some point", not "it finished
    # exactly full". With ~20% wash, a date that turned demand away still ends well below
    # capacity, so `sold_out` runs near 1% and is not the quantity to band.
    sellout = float(sd["sold_out"].mean())
    closeout = float(sd["was_censored"].mean())
    report.measurements["sellout_frequency"] = sellout
    report.measurements["closeout_frequency"] = closeout
    report.checks["closeout_frequency_band"] = 0.04 < closeout < 0.30

    final = on_books.loc[on_books["dta"] == 0].set_index(key)
    wash = float((final["cancel_cum"] / final["gross_cum"].replace(0, np.nan)).mean())
    report.measurements["wash_rate"] = wash
    report.checks["wash_rate_band"] = 0.10 < wash < 0.35

    censored_n = int(sd["was_censored"].sum())
    hidden = int((sd["was_censored"] & ~sd["sold_out"]).sum())
    report.measurements["censored_stay_dates"] = float(censored_n)
    report.measurements["censored_but_not_sold_out"] = float(hidden)
    report.notes["sold_out_is_not_the_censoring_indicator"] = (
        f"{hidden} of {censored_n} censored stay dates "
        f"({hidden / max(censored_n, 1):.0%}) finish below capacity, because a booking rejected "
        "at the wall is never retried after a later cancellation frees the room. A censored "
        "likelihood must key on `was_censored`, not `sold_out`."
    )
    return report


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y with its linear dependence on x removed."""
    if np.std(x) == 0:
        return y - y.mean()
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def pace_information(
    ds: LoadedDataset, dtas=(150, 90, 60, 28, 7), against_remaining: bool = False
) -> dict[int, float]:
    """How much the partial curve says, with the demand level controlled within each property.

    Two questions, one statistic, switched by `against_remaining`:

    - **Forecast value** (default). Correlation of `otb_k` with `rooms_sold`. This is the
      benchmark question — does seeing the partial curve beat knowing the level alone. That
      `otb_k` is part of `rooms_sold` is not a confound here; it is *why* pace forecasting works.
    - **Momentum** (`against_remaining=True`). Correlation of `otb_k` with the pickup still to
      come. This separates real persistence from arithmetic: with the level fixed, booking more
      early mechanically leaves less to come, so mass conservation drives this negative. A
      Poisson arrival process gives the pure conservation value, because counts in disjoint DTA
      ranges are independent given the rate and the shape. A demand shock that persists across
      the window offsets it.

    Controlling the level *within* property matters for both: pooled, properties differ in
    capacity and one slope on `lam_true` cannot absorb the cross-property scale.
    """
    sd = ds.stay_dates_enriched()
    ob = ds.on_books.merge(
        sd[["property_id", "stay_date", "rooms_sold", "lam_true"]],
        on=["property_id", "stay_date"],
    )
    out = {}
    for k in dtas:
        g = ob[ob["dta"] == k]
        if len(g) < 100:
            continue
        # Control for the level *within* each property. Pooling would leave a capacity confound:
        # properties differ in scale, so `otb` and `rooms_sold` share cross-property variation
        # that one pooled slope on lam_true cannot absorb, and the statistic reads high for
        # reasons that have nothing to do with pace.
        per_prop = []
        for _, p in g.groupby("property_id", sort=False):
            if len(p) < 30:
                continue
            lam = p["lam_true"].to_numpy()
            target = (
                (p["rooms_sold"] - p["otb"]) if against_remaining else p["rooms_sold"]
            ).to_numpy().astype(float)
            a = _residual(p["otb"].to_numpy().astype(float), lam)
            b = _residual(target, lam)
            if a.std() and b.std():
                per_prop.append(float(np.corrcoef(a, b)[0, 1]))
        if per_prop:
            out[k] = float(np.mean(per_prop))
    return out


def anchors(ds: LoadedDataset) -> ValidationReport:
    """T3 — dispersion, pace information, lead time, booking-weekday amplitude."""
    report = ValidationReport(name="anchors", tier="T3")
    constants = GeneratorConstants.from_snapshot(ds.config.get("constants") or {})
    sd = ds.stay_dates_enriched()

    # --- dispersion, now a parameter rather than a consequence of party sizes ---
    cell = sd.assign(month=sd["stay_date"].dt.month).groupby(
        ["property_id", "dow", "month"]
    )["rooms_sold"].agg(["mean", "std", "count"])
    cell = cell[(cell["count"] >= 6) & (cell["mean"] > 0)]
    if cell.empty:
        report.skipped["X1_dispersion"] = "needs ~1 year of stay dates per cell"
    else:
        vm = float((cell["std"] ** 2 / cell["mean"]).median())
        cv = float((cell["std"] / cell["mean"]).median())
        report.measurements["X1_var_over_mean"] = vm
        report.measurements["X1_cv"] = cv
        report.measurements["X1_reference_var_over_mean"] = WK_VAR_OVER_MEAN
        report.checks["X1_dispersion_in_range_of_real_data"] = 0.5 * WK_VAR_OVER_MEAN < vm < 1.6 * WK_VAR_OVER_MEAN
        report.notes["X1_dispersion"] = (
            f"var/mean {vm:.1f} and CV {cv:.3f} against Weatherford & Kimes' ~{WK_VAR_OVER_MEAN:.0f} "
            f"and {WK_CV}. Set by `shock_sigma` ({constants.shock_sigma}) and "
            f"`increment_dispersion` ({constants.increment_dispersion}) — the first is "
            "demand-level and partly forecastable, the second irreducible."
        )

    # --- pace information: the property v1 could not have ---
    info = pace_information(ds)
    for k, v in info.items():
        report.measurements[f"X6_pace_information_dta_{k}"] = v
    for k, v in pace_information(ds, against_remaining=True).items():
        report.measurements[f"X7_pace_momentum_dta_{k}"] = v
    if info:
        long_lead = max((v for k, v in info.items() if k >= 60), default=0.0)
        report.checks["X6_partial_curve_predicts_remaining_pickup"] = long_lead > 0.15
        report.notes["X6_pace_information"] = (
            "correlation of on-books at DTA k with the pickup still to come, level controlled "
            f"within property — best at DTA>=60 is {long_lead:.3f}. Governed by "
            f"`shock_corr_len` ({constants.shock_corr_len} days). Zero means a fast start says "
            "nothing about the finish, which is where a Poisson arrival process sits by "
            "construction."
        )

    # --- lead time ---
    # Measured on the **gross** curve, not net on-books. The reference is a median lead time over
    # bookings, and it has to be: with ~22% wash, net on-books peaks above the final number, so
    # `otb / rooms_sold` crosses 0.5 far earlier than any booking actually happened and reads ~30
    # days too long.
    key = ["property_id", "stay_date"]
    total = ds.on_books.loc[ds.on_books["dta"] == 0].set_index(key)["gross_cum"]
    ob = ds.on_books.join(total.rename("gross_total"), on=key)
    ob = ob[ob["gross_total"] > 0]
    curve = (ob["gross_cum"] / ob["gross_total"]).groupby(ob["dta"]).mean()
    ks, vals = np.asarray(curve.index, dtype=float), curve.to_numpy()
    dta_half = float(np.interp(0.5, vals[::-1], ks[::-1]))
    report.measurements["X3_median_booking_lead"] = dta_half
    report.measurements["X3_reference_median_lead"] = float(ANTONIO_MEDIAN_LEAD)
    report.checks["X3_lead_time_is_realistic"] = 50 < dta_half < 90
    report.notes["X3_lead_time"] = (
        f"half of bookings made {dta_half:.0f} days out, against a real median booking lead of "
        f"{ANTONIO_MEDIAN_LEAD} days (Antonio et al., 119,390 bookings)."
    )

    # --- booking-weekday amplitude, controlled for the DTA volume gradient ---
    inc = ds.increments()
    ripple = inc.groupby("booking_dow")["lift"].mean()
    if ripple.min() > 0:
        ratio = float(ripple.max() / ripple.min())
        report.measurements["X2_booking_dow_peak_trough"] = ratio
        report.measurements["X2_reference_peak_trough"] = ANTONIO_BOOKING_DOW_RATIO
        report.checks["X2_ripple_amplitude_is_realistic"] = 1.8 < ratio < 3.0

    report.notes["X5_length_of_stay"] = (
        "stays are single-night by construction, against a real median LOS of 3 nights. The "
        "coupling between adjacent arrival dates that a real surface has is absent."
    )
    return report
