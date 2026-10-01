"""The booking-pace surface itself: a 3D (arrival_date, DTA) -> OTB plot, read off a dataset.

Spec: docs/simulation-spec.md; this is the object docs/design.md's headline claim is about —
"a 2D booking-pace surface indexed by (arrival_date, DTA)" — rendered directly rather than
flattened into a 1D time series.

Needs an actual dataset, like `calendar_figures.py` and unlike `figures.py`.

Rendered by `scripts/plot_booking_surface.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.figures import INK, INK_2, SURFACE, Figure, _header, _save, _style


def fig_booking_surface(
    ds: LoadedDataset,
    out_path: Path,
    property_id: int | None = None,
    arrival_start: str | None = None,
    arrival_end: str | None = None,
    dta_max: int = 180,
) -> Figure:
    """The (arrival_date, DTA) -> OTB surface: portfolio-wide by default, or one property.

    A single property's curve at long DTA is mostly the per-(property, stay_date) shock draw, not
    the systematic shape — v100 in particular predates `shock_corr_len` (design.md §12, the
    2026-09-09 entry), so one property's curve at DTA 90 swings from 0 to near-full night over
    night. Summing OTB across the portfolio for each (arrival_date, DTA) cell averages that
    independent per-property noise out by the same logic `calendar_figures._booking_date_residual`
    uses across properties, and leaves the systematic level/seasonality/DOW shape visible.
    """
    ob = ds.on_books
    sub = ob.copy()
    sub["stay_date"] = pd.to_datetime(sub["stay_date"])
    if property_id is not None:
        sub = sub[sub["property_id"] == property_id]
    if arrival_start is not None:
        sub = sub[sub["stay_date"] >= pd.Timestamp(arrival_start)]
    if arrival_end is not None:
        sub = sub[sub["stay_date"] <= pd.Timestamp(arrival_end)]
    sub = sub[sub["dta"] <= dta_max]

    agg = "sum" if property_id is None else "first"
    grid = sub.pivot_table(index="stay_date", columns="dta", values="otb",
                            aggfunc=agg).sort_index()
    grid = grid.reindex(columns=sorted(grid.columns))
    arrivals = grid.index
    dtas = grid.columns.to_numpy(dtype=float)

    plt = _style()
    fig = plt.figure(figsize=(10, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)

    # DTA decreasing left-to-right (arrival at the front), matching the reference orientation
    X = np.tile(np.arange(len(arrivals))[:, None], (1, len(dtas)))
    Y = np.tile(dtas[None, :], (len(arrivals), 1))
    Z = grid.to_numpy()

    ax.plot_surface(X, Y, Z, cmap="Blues", edgecolor="none", antialiased=True,
                     rcount=len(arrivals), ccount=len(dtas))

    n_ticks = min(8, len(arrivals))
    tick_idx = np.linspace(0, len(arrivals) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([arrivals[i].strftime("%b %-d") for i in tick_idx], rotation=20,
                        ha="right", fontsize=7.5, color=INK_2)
    ax.set_ylim(dta_max, 0)
    ax.set_xlabel("Arrival Date", color=INK_2, labelpad=14)
    ax.set_ylabel("DTA", color=INK_2, labelpad=8)
    ax.set_zlabel("OTB", color=INK_2, labelpad=6)
    ax.tick_params(colors=INK_2, labelsize=7.5)
    ax.view_init(elev=22, azim=-60)
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)

    if property_id is None:
        who = f"portfolio sum, {len(ds.properties)} properties"
    else:
        props = ds.properties.set_index("property_id")
        who = f"property {property_id} ({props.loc[property_id, 'archetype']}, " \
              f"capacity {int(props.loc[property_id, 'capacity'])})"
    _header(
        fig, "The booking-pace surface",
        f"{ds.root.name}, {who} — rooms on the books by (arrival date, days-to-arrival), "
        f"{arrivals.min():%Y-%m-%d} to {arrivals.max():%Y-%m-%d}",
        top=0.92,
    )
    return _save(fig, out_path, f"The booking-pace surface ({who}) in {ds.root.name}.")
