"""Calendar effects recovered *from a generated dataset*, blind — no ground-truth calendar read.

Spec: docs/design.md §2.3 (the diagonal claim); docs/validation-design.md §3.4-3.5 (M4, D0-D3);
the planted effects themselves are `calendars.py` (`build_promo_calendar`, `GeneratorConstants
.book_dow`).

Unlike `figures.py`, this needs an actual dataset: the claim under test is what the *observed*
booking-pace surface looks like, not what the generator's functions say it should. Detection is
blind by design — it never reads `promo_calendar.parquet` — so it renders on datasets that
predate that file (e.g. `_data/v100`, from the archived `deeppace_sim_v1` generator) the same way
it would on a current one. It is the same residual statistic
`deeppace_sim_v1.diagnostics.detect_diagonals` uses for D0/D1/D3, reused here for a picture
instead of a pass/fail.

Two effects, one mechanism: `booking_date = arrival_date - DTA`, so anything that varies by
booking date (the weekly book_dow ripple, a flash-sale promo) touches every arrival date at a
*different* DTA and reads as a diagonal streak in the surface, not a per-cell feature.

Rendered by `scripts/plot_calendar_effects.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.figures import (
    AXIS,
    BLUE,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    Figure,
    _header,
    _save,
    _style,
)


def _booking_date_residual(inc: pd.DataFrame) -> pd.Series:
    """Mean lift per booking date, with the booking-DOW ripple divided back out.

    Portfolio-wide average (suppresses per-property noise; a promo is portfolio-wide) divided by
    its own weekday's mean (isolates a date-specific anomaly from the ordinary weekly ripple).
    Matches `deeppace_sim_v1.diagnostics._booking_date_residual`, the statistic behind D1.
    """
    by_date = inc.groupby("booking_date")["lift"].mean()
    dow = by_date.index.weekday
    return by_date / by_date.groupby(dow).transform("mean")


def _cell_lift(inc: pd.DataFrame) -> pd.Series:
    """Per-cell activity relative to its (dta, booking weekday) baseline — the D3 statistic."""
    baseline = inc.groupby(["dta", "booking_dow"])["gross_inc"].transform("mean")
    return inc["gross_inc"] / baseline.replace(0, np.nan)


def fig_calendar_effects(ds: LoadedDataset, out_path: Path, top_n: int = 15) -> Figure:
    """Three views of the same diagonal claim, all read off the surface with no ground truth.

    (1) booking-DOW ripple — mean lift by booking weekday, portfolio-wide.
    (2) the booking-date residual across the whole window, spiking on the anomalous dates.
    (3) a zoom on the sharpest cluster: the (arrival_date, DTA) grid itself, where the anomaly
        traces a clean diagonal rather than a single cell or a whole row/column.
    """
    plt = _style()
    inc = ds.increments()
    inc = inc.assign(cell_lift=_cell_lift(inc))

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], width_ratios=[1, 2.4],
                          hspace=0.42, wspace=0.28)

    # --- 1: booking-DOW ripple ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ripple = inc.groupby("booking_dow")["lift"].mean()
    ripple = ripple.reindex(range(7))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colours = [ORANGE if v == ripple.max() else (MUTED if v == ripple.min() else BLUE)
               for v in ripple]
    ax.bar(labels, ripple.to_numpy(), color=colours, width=0.7)
    ax.axhline(ripple.mean(), color=AXIS, lw=1, ls=(0, (4, 3)))
    peak_trough = float(ripple.max() / ripple.min())
    ax.set_title(f"1 · booking-DOW ripple  (peak/trough {peak_trough:.2f}x)",
                 loc="left", color=INK, fontsize=10)
    ax.set_ylabel("mean lift, by booking weekday")
    ax.grid(axis="y")

    # --- 2: booking-date residual, whole window -----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    resid = _booking_date_residual(inc).sort_index()
    flagged = resid.sort_values(ascending=False).head(top_n)
    ax.plot(resid.index, resid.to_numpy(), color=MUTED, lw=0.9)
    ax.scatter(flagged.index, flagged.to_numpy(), color=ORANGE, s=22, zorder=3)
    ax.axhline(1.0, color=AXIS, lw=1)
    top3 = flagged.sort_values(ascending=False).head(3)
    for d, v in top3.items():
        ax.annotate(d.strftime("%Y-%m-%d"), (d, v), xytext=(0, 7), textcoords="offset points",
                    ha="center", color=ORANGE, fontsize=7.5)
    ax.set_title(f"2 · booking-date residual — top {top_n} flagged, blind (no promo calendar read)",
                 loc="left", color=INK, fontsize=10)
    ax.set_ylabel("lift ÷ that weekday's mean")
    ax.grid(axis="y")

    # --- 3: the diagonal itself, zoomed on the sharpest cluster --------------------------------
    ax = fig.add_subplot(gs[1, :])
    # late Nov - mid Jan carries its own "year-end structure" strong enough to dominate this
    # statistic on its own (validation-design.md §3.5); excluded here so the zoom shows an
    # isolated promo diagonal rather than that confound.
    is_year_end = (flagged.index.month == 12) | (flagged.index.month == 1) | (
        (flagged.index.month == 11) & (flagged.index.day >= 15))
    candidates = flagged[~is_year_end]
    candidates = candidates if len(candidates) else flagged
    cluster_start = pd.Timestamp(candidates.sort_values(ascending=False).index[0])
    win_start = cluster_start - timedelta(days=2)
    win_end = cluster_start + timedelta(days=19)
    mask = (inc["stay_date"] >= win_start) & (inc["stay_date"] <= win_end)
    grid = (inc[mask].groupby(["stay_date", "dta"])["cell_lift"].mean()
            .unstack("dta").sort_index())
    grid = grid.reindex(columns=range(int(inc["dta"].max()) + 1))

    vmax = float(np.nanmax(grid.to_numpy()))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("diag", [BLUE, SURFACE, ORANGE])
    im = ax.imshow(grid.T.to_numpy(), aspect="auto", cmap=cmap, vmin=2 - vmax, vmax=vmax,
                   origin="lower", extent=(-0.5, len(grid.index) - 0.5, -0.5, grid.shape[1] - 0.5))
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
    cbar.set_label("cell lift, × its (DTA, booking-weekday) baseline", color=INK_2, fontsize=8)
    cbar.ax.tick_params(labelsize=7.5, colors=MUTED)

    # trace the guide diagonal for each flagged booking date inside this window
    promo_days = [d for d in flagged.index if win_start <= d <= win_end]
    for i, bd in enumerate(sorted(promo_days)):
        xs, ys = [], []
        for j, sd in enumerate(grid.index):
            k = (sd - bd).days
            if 0 <= k <= grid.shape[1] - 1:
                xs.append(j)
                ys.append(k)
        if xs:
            ax.plot(xs, ys, color=INK, lw=1.0, ls=(0, (1, 1.4)), alpha=0.75)
            ax.annotate(f"booking_date {bd.strftime('%Y-%m-%d')}", (xs[-1], ys[-1]),
                        xytext=(6, 9 * i - 5), textcoords="offset points", va="center",
                        color=INK, fontsize=7.5)

    ax.set_xticks(range(len(grid.index)))
    ax.set_xticklabels([d.strftime("%m-%d") for d in grid.index], rotation=60, fontsize=7,
                       color=MUTED)
    ax.set_yticks(range(0, grid.shape[1], 2))
    ax.set_xlabel("arrival date")
    ax.set_ylabel("DTA")
    ax.set_title("3 · the same anomaly in (arrival_date, DTA) space — a diagonal, not a cell or a row",
                 loc="left", color=INK, fontsize=10)

    _header(
        fig, "Calendar effects live on the booking-date diagonal",
        f"{ds.root.name}: DOW ripple {peak_trough:.2f}x peak/trough; the "
        f"{cluster_start:%Y-%m-%d} cluster traces a clean diagonal, invisible to any "
        "per-(arrival_date, DTA) feature — both recovered with no calendar file read",
        top=0.90,
    )
    return _save(fig, out_path, f"Calendar effects on {ds.root.name}: DOW ripple, blind "
                 "booking-date residual, and the recovered diagonal.")
