"""Origin-based, leakage-safe inputs for the two-stage final rooms-sold forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deeppace_net.foundation import calendar
from deeppace_sim.dataset import LoadedDataset


@dataclass(frozen=True)
class PaceBatch:
    rows: pd.DataFrame
    curves: np.ndarray  # N x (K+1) x 3: OTB/capacity, net change/capacity, DTA/K
    observed: np.ndarray
    shared: np.ndarray  # booking-date sequence: portfolio activity, coverage, weekday
    context: np.ndarray  # horizon/K and arrival calendar


class PaceData:
    """Index the surface once. Ground-truth latents are never used as inputs or targets."""

    def __init__(self, ds: LoadedDataset, property_ids: list[int] | None = None):
        if not ds.full_grid:
            raise ValueError("Two-stage pace currently requires the full daily DTA grid")
        ids = property_ids if property_ids is not None else ds.properties.property_id.tolist()
        self.properties = ds.properties[ds.properties.property_id.isin(ids)].copy()
        if self.properties.empty:
            available = ds.properties.property_id.tolist()
            raise ValueError(
                f"No properties matched property_ids={ids!r}. "
                f"Dataset {ds.root} has {len(available)} properties, "
                f"e.g. {available[:5]}."
            )
        self.history = ds.stay_dates.loc[
            ds.stay_dates.property_id.isin(ids), ["property_id", "stay_date", "rooms_sold"]
        ].copy()
        self.history["stay_date"] = pd.to_datetime(self.history.stay_date)
        ob = ds.on_books[ds.on_books.property_id.isin(ids)].copy()
        ob["stay_date"] = pd.to_datetime(ob.stay_date)
        ob["booking_date"] = pd.to_datetime(ob.booking_date)
        if not (ob.booking_date == ob.stay_date - pd.to_timedelta(ob.dta, unit="D")).all():
            raise ValueError("booking_date must equal stay_date - DTA")
        self.k = int(ob.dta.max())
        self.surface = ob.set_index(["property_id", "stay_date", "dta"])["otb"]
        if self.surface.index.has_duplicates:
            raise ValueError("Duplicate surface cells")
        caps = self.properties.set_index("property_id").capacity
        ob = ob.sort_values(["property_id", "stay_date", "dta"], ascending=[True, True, False])
        # Opening balances are levels, not fabricated booking increments.
        ob["net"] = ob.groupby(["property_id", "stay_date"]).otb.diff()
        ob["net"] /= ob.property_id.map(caps)
        daily = ob.groupby("booking_date").net.agg(["mean", "count"])
        self.daily = daily.rename(columns={"mean": "net", "count": "coverage"})

    def build(
        self,
        as_of: str | pd.Timestamp,
        horizons: tuple[int, ...],
        shared_days: int = 90,
        labeled: bool = True,
    ) -> PaceBatch:
        t = pd.Timestamp(as_of)
        if not horizons or min(horizons) < 1 or max(horizons) > self.k:
            raise ValueError(f"horizons must be within 1..{self.k}")
        if shared_days < 1:
            raise ValueError("shared_days must be positive")
        grid = np.arange(self.k, -1, -1)
        rows, curves, masks = [], [], []
        targets = self.history.set_index(["property_id", "stay_date"]).rooms_sold
        for prop in self.properties.itertuples():
            for h in horizons:
                d = t + pd.Timedelta(days=h)
                mask = grid >= h
                # Reindex only observable cells; future values never enter the feature arrays.
                keys = pd.MultiIndex.from_tuples(
                    [(prop.property_id, d, int(k)) for k in grid[mask]]
                )
                observed = self.surface.reindex(keys).to_numpy(dtype=np.float32)
                if not np.isfinite(observed).all():
                    raise ValueError(f"Missing observed OTB for property {prop.property_id}, {d}")
                values = np.zeros(len(grid), dtype=np.float32)
                values[mask] = observed / prop.capacity
                delta = np.zeros_like(values)
                delta[1 : len(observed)] = np.diff(observed) / prop.capacity
                curves.append(np.stack([values, delta, grid / self.k], axis=-1))
                masks.append(mask)
                row = dict(
                    property_id=prop.property_id,
                    as_of=t,
                    stay_date=d,
                    horizon=h,
                    capacity=prop.capacity,
                    archetype=prop.archetype,
                    current_otb=observed[-1],
                )
                if labeled:
                    row["target"] = float(targets.loc[(prop.property_id, d)])
                rows.append(row)
        table = pd.DataFrame(rows)
        context = calendar(pd.DatetimeIndex(table.stay_date)).drop(columns="timestamp")
        context.insert(0, "horizon", table.horizon.to_numpy() / self.k)
        dates = pd.date_range(end=t, periods=shared_days)
        daily = self.daily.reindex(dates)
        shared = np.stack(
            [
                daily.net.fillna(0),
                np.log1p(daily.coverage.fillna(0)),
                np.sin(2 * np.pi * dates.dayofweek / 7),
                np.cos(2 * np.pi * dates.dayofweek / 7),
            ],
            axis=-1,
        )
        return PaceBatch(
            table,
            np.asarray(curves, dtype=np.float32),
            np.asarray(masks),
            shared.astype(np.float32),
            context.to_numpy(dtype=np.float32),
        )
