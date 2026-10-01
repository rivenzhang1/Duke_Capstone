"""Reading a generated dataset back off disk — the one loader every check shares.

Spec: docs/simulation-spec.md §2 (schema), docs/validation-design.md §6.

Exists so that no check re-derives ground truth by replaying the generator. A check that cannot
find its ground truth here should report `n/a`, not reconstruct it — reconstruction couples the
check to the generator's stream layout and breaks silently when that changes.

Usage:
    ds = LoadedDataset.load("_data/v3")
    ds.stay_dates, ds.ground_truth, ds.promo_calendar
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ground-truth columns added for the validation tiers; absent in datasets before v2
_LATENT_COLUMNS = ("event_mult", "mood", "gross_drawn", "shock_total")


@dataclass(frozen=True)
class LoadedDataset:
    root: Path
    config: dict
    properties: pd.DataFrame
    stay_dates: pd.DataFrame
    on_books: pd.DataFrame
    ground_truth: pd.DataFrame
    promo_calendar: pd.DataFrame | None
    event_calendar: pd.DataFrame | None
    full_grid: bool
    """True when `on_books` is the complete 0..k_max grid rather than the bucketed one.

    Increment-based checks (the booking-DOW ripple, the promo diagonals) need consecutive DTA to
    difference the cumulative columns. On the bucketed grid only the dense `dta <= 14` region
    qualifies, which limits how much of a promo's effect is visible.
    """

    @property
    def has_latents(self) -> bool:
        return all(c in self.ground_truth.columns for c in _LATENT_COLUMNS)

    @property
    def generator_hash(self) -> str:
        return self.config.get("generator_hash", "unknown")

    @property
    def dataset_version(self) -> str:
        return self.config.get("dataset_version", "unknown")

    @property
    def capacity(self) -> pd.Series:
        return self.properties.set_index("property_id")["capacity"]

    @property
    def archetype(self) -> pd.Series:
        return self.properties.set_index("property_id")["archetype"]

    def stay_dates_enriched(self) -> pd.DataFrame:
        """`stay_dates` + archetype, capacity, occupancy, and `occ_expected`.

        `occ_expected` is the ex-ante fullness proxy that drives compression and the mixture
        weights. Not persisted because it is exactly recoverable from what is.
        """
        sd = self.stay_dates.copy()
        sd["stay_date"] = pd.to_datetime(sd["stay_date"])
        sd["archetype"] = sd["property_id"].map(self.archetype)
        sd["capacity"] = sd["property_id"].map(self.capacity)
        sd["occ"] = sd["rooms_sold"] / sd["capacity"]
        wash = float(self.config.get("constants", {}).get("expected_wash", 0.0))
        sd["occ_expected"] = (sd["lam_true"] * (1.0 - wash) / sd["capacity"]).clip(0.0, 1.3)
        return sd

    def increments(self, max_dta: int | None = None) -> pd.DataFrame:
        """Per-cell booking activity: differences of the reverse-cumulative columns.

        `gross_cum` is cumulative *backwards* in DTA, so activity on the booking date at DTA k is
        ``gross_cum[k] - gross_cum[k+1]``. On the bucketed grid this is only meaningful where the
        grid is consecutive, so `max_dta` defaults to 14 there and to the whole grid otherwise.

        The largest DTA in each group has no successor to difference against, so its row is
        **dropped**, not zero-filled. Zero-filling fabricates a no-activity cell whose booking
        weekday varies with the stay date, which biases any per-booking-weekday statistic.
        Dropping it also leaves the bucketed window at dta 0..13 — exactly two weeks, so each
        booking weekday is represented equally.

        The returned `lift` column is each cell's activity relative to the mean at its own DTA.
        Booking volume rises steeply as DTA falls, and which DTA maps to which booking weekday
        depends on the stay date's own weekday, so a raw mean by booking weekday carries a
        DTA-volume confound: with the ripple switched off entirely it still reads 1.11
        peak-to-trough, against 1.04 for the DTA-controlled version. Prefer `lift` for any
        per-booking-weekday or per-booking-date statistic.
        """
        if max_dta is None:
            max_dta = None if self.full_grid else 14
        ob = self.on_books
        if max_dta is not None:
            ob = ob[ob["dta"] <= max_dta]
        ob = ob.sort_values(["property_id", "stay_date", "dta"], ascending=[True, True, False])
        key = ["property_id", "stay_date"]
        out = ob.copy()
        out["gross_inc"] = out.groupby(key, sort=False)["gross_cum"].diff()
        out["cancel_inc"] = out.groupby(key, sort=False)["cancel_cum"].diff()
        out = out.dropna(subset=["gross_inc"])
        out["booking_dow"] = out["booking_date"].dt.weekday
        by_dta = out.groupby("dta")["gross_inc"].transform("mean")
        out["lift"] = out["gross_inc"] / by_dta.replace(0, np.nan)
        return out

    @classmethod
    def load(cls, root: str | Path, full_grid: bool | None = None) -> LoadedDataset:
        root = Path(root)

        def maybe(name: str) -> pd.DataFrame | None:
            path = root / name
            return pd.read_parquet(path) if path.exists() else None

        use_full = (root / "on_books.parquet").exists() if full_grid is None else full_grid
        on_books = pd.read_parquet(root / ("on_books.parquet" if use_full else "on_books_bucketed.parquet"))
        return cls(
            root=root,
            config=json.loads((root / "config.json").read_text()),
            properties=pd.read_parquet(root / "properties.parquet"),
            stay_dates=pd.read_parquet(root / "stay_dates.parquet"),
            on_books=on_books,
            ground_truth=pd.read_parquet(root / "ground_truth.parquet"),
            promo_calendar=maybe("promo_calendar.parquet"),
            event_calendar=maybe("event_calendar.parquet"),
            full_grid=use_full,
        )
