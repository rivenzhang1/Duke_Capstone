"""Parquet writers — owns the on-disk schema.

Spec: docs/simulation-spec.md.

Outputs, under ``_data/<dataset_version>/``:

    properties.parquet           one row per property
    stay_dates.parquet           n_props x n_days — includes the target `rooms_sold`
    on_books.parquet             x(k_max+1) the above, the full DTA grid (~400 MB at 60 props)
    on_books_bucketed.parquet    the 28-point bucketed grid — use this for iteration
    ground_truth.parquet         every latent, one row per (property, stay date)
    promo_calendar.parquet       the planted flash-sale booking dates + multipliers
    event_calendar.parquet       per-property event dates + multipliers
    config.json                  the full SimConfig incl. every constant, plus generator_hash

`booking_date` is **materialised** on the on-books tables rather than derived at read time. It
makes diagonal analysis trivial and it is what the as-of leakage rule slices on.

## What counts as ground truth

Every quantity the generator chose, not merely the ones cheap to store. The pace shock is the
one that needs a decision: the field is a value per (stay date, DTA) cell, which is the same size
as the surface itself. It is persisted in two recoverable forms instead — `shock_total`, the
realised demand as a multiple of `lam_true`, and `shock_band_*`, its mean over four DTA bands.
That is enough for the diagnostic that matters: whether a model's learned pace state correlates
with the deviation actually applied, rather than merely predicting well.

`generator_hash` digests every constant the generator reads. `dataset_version` names an intent;
the hash names the generator, so two datasets sharing a version string but not a generator cannot
be silently compared.

Public surface:
    write_dataset(portfolio, config, out_dir, bucketed_only=False) -> DatasetPaths
    bucket_on_books(df, dta_grid) -> pd.DataFrame
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_sim.config import SimConfig
from deeppace_sim.generate import PropertyState, StayDateResult, promo_calendar

SHOCK_BAND_NAMES = ("shock_band_0_7", "shock_band_8_30", "shock_band_31_90", "shock_band_91_up")


@dataclass(frozen=True)
class DatasetPaths:
    properties: Path
    stay_dates: Path
    on_books_bucketed: Path
    ground_truth: Path
    promo_calendar: Path
    event_calendar: Path
    config: Path
    on_books: Path | None = None


def bucket_on_books(df: pd.DataFrame, dta_grid: tuple[int, ...]) -> pd.DataFrame:
    return df[df["dta"].isin(dta_grid)].reset_index(drop=True)


def _config_to_json(config: SimConfig) -> dict:
    d = asdict(config)
    d["start"] = config.start.isoformat()
    d["end"] = config.end.isoformat()
    d["as_of"] = config.as_of.isoformat()
    d["output_dir"] = str(config.output_dir)
    d["generator_hash"] = config.generator_hash
    return d


def write_dataset(
    portfolio: list[tuple[PropertyState, list[StayDateResult]]],
    config: SimConfig,
    out_dir: Path,
    bucketed_only: bool = False,
) -> DatasetPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prop_rows, stay_rows, gt_rows, event_rows = [], [], [], []
    pid_col, stay_col, dta_col = [], [], []
    otb_col, gross_col, cancel_col, adr_col = [], [], [], []

    for prop, results in portfolio:
        prop_rows.append(dict(
            property_id=prop.property_id, archetype=prop.archetype,
            capacity=prop.capacity, base_adr=prop.base_adr, base_demand=prop.base_demand,
        ))
        for ev_date, mult in sorted(prop.event_calendar.items()):
            event_rows.append(
                dict(property_id=prop.property_id, stay_date=ev_date, event_mult=mult)
            )
        for r in results:
            stay_rows.append(dict(
                property_id=r.property_id, stay_date=r.stay_date, dow=r.dow,
                rooms_sold=r.rooms_sold, unconstrained_demand=r.unconstrained_demand,
                turned_away=r.turned_away, sold_out=r.sold_out, was_censored=r.was_censored,
                lam_true=r.lam_true, is_event=r.is_event, is_holiday=r.is_holiday,
                w_group=r.weights[0], w_leisure=r.weights[1], w_lastmin=r.weights[2],
            ))
            gt_rows.append(dict(
                property_id=r.property_id, stay_date=r.stay_date, lam_true=r.lam_true,
                w_group=r.weights[0], w_leisure=r.weights[1], w_lastmin=r.weights[2],
                event_mult=r.event_mult, mood=r.mood,
                gross_drawn=r.gross_drawn, shock_total=r.shock_total,
                **dict(zip(SHOCK_BAND_NAMES, r.shock_bands, strict=True)),
                unconstrained_demand=r.unconstrained_demand, turned_away=r.turned_away,
                was_censored=r.was_censored,
            ))

            n_k = len(r.otb)
            pid_col.append(np.full(n_k, r.property_id, dtype=np.int32))
            stay_col.append(np.full(n_k, np.datetime64(r.stay_date), dtype="datetime64[D]"))
            dta_col.append(np.arange(n_k, dtype=np.int16))
            otb_col.append(r.otb.astype(np.int32))
            gross_col.append(r.gross_cum.astype(np.int32))
            cancel_col.append(r.cancel_cum.astype(np.int32))
            adr_col.append(r.adr_by_k.astype(np.float32))

    properties_df = pd.DataFrame(prop_rows)
    stay_dates_df = pd.DataFrame(stay_rows)
    ground_truth_df = pd.DataFrame(gt_rows)
    # keep stay_date a real datetime in every table so they join without coercion
    stay_dates_df["stay_date"] = pd.to_datetime(stay_dates_df["stay_date"])
    ground_truth_df["stay_date"] = pd.to_datetime(ground_truth_df["stay_date"])

    on_books_df = pd.DataFrame({
        "property_id": np.concatenate(pid_col),
        "stay_date": np.concatenate(stay_col),
        "dta": np.concatenate(dta_col),
        "otb": np.concatenate(otb_col),
        "gross_cum": np.concatenate(gross_col),
        "cancel_cum": np.concatenate(cancel_col),
        "adr": np.concatenate(adr_col),
    })
    on_books_df["booking_date"] = on_books_df["stay_date"] - pd.to_timedelta(
        on_books_df["dta"], unit="D"
    )

    promo = promo_calendar(config)
    promo_df = pd.DataFrame({
        "booking_date": pd.to_datetime(sorted(promo)),
        "promo_mult": [promo[d] for d in sorted(promo)],
    })
    event_df = pd.DataFrame(event_rows, columns=["property_id", "stay_date", "event_mult"])
    if not event_df.empty:
        event_df["stay_date"] = pd.to_datetime(event_df["stay_date"])

    paths = dict(
        properties=out_dir / "properties.parquet",
        stay_dates=out_dir / "stay_dates.parquet",
        ground_truth=out_dir / "ground_truth.parquet",
        on_books_bucketed=out_dir / "on_books_bucketed.parquet",
        promo_calendar=out_dir / "promo_calendar.parquet",
        event_calendar=out_dir / "event_calendar.parquet",
        config=out_dir / "config.json",
    )
    properties_df.to_parquet(paths["properties"], index=False)
    stay_dates_df.to_parquet(paths["stay_dates"], index=False)
    ground_truth_df.to_parquet(paths["ground_truth"], index=False)
    promo_df.to_parquet(paths["promo_calendar"], index=False)
    event_df.to_parquet(paths["event_calendar"], index=False)
    bucket_on_books(on_books_df, config.dta_grid).to_parquet(
        paths["on_books_bucketed"], index=False
    )

    on_books_path = None
    if not bucketed_only:
        on_books_path = out_dir / "on_books.parquet"
        on_books_df.to_parquet(on_books_path, index=False)

    paths["config"].write_text(json.dumps(_config_to_json(config), indent=2))
    return DatasetPaths(on_books=on_books_path, **paths)
