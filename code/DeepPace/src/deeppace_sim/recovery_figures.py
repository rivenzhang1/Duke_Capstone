"""Plots of the T1 parameter-recovery checks in `recovery.py` — one figure per register entry.

Needs an actual dataset, like `calendar_figures.py` and `surface_figures.py`. Every number here
comes from the same functions `recovery.py`'s checks assert against, so a figure and its check
can never quietly disagree about what "recovered" means.

Rendered by `scripts/plot_recovery.py` into `reports/figures/recovery/`, and referenced by
`reports/parameter_recovery_report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deeppace_sim.config import ARCHETYPES, GeneratorConstants
from deeppace_sim.calendars import holiday_dates
from deeppace_sim.dataset import LoadedDataset
from deeppace_sim.figures import AXIS, BLUE, INK, INK_2, MUTED, ORANGE, SURFACE, Figure, _header, _save, _style
from deeppace_sim.recovery import (
    SHOCK_BANDS,
    _isolate,
    _sd_with_latents,
    _tol,
    lam_residuals,
    shape_curves,
    shock_band_correlations,
)

ARCHETYPE_COLOURS = {
    "urban_business": BLUE, "resort": ORANGE, "airport": "#1baf7a",
    "convention": "#eda100", "suburban_select": "#e87ba4",
}


def fig_lam_residual(ds: LoadedDataset, out_path: Path) -> Figure:
    """R1: the lam residual — a histogram against its configured noise, and "no structure left"."""
    plt = _style()
    constants = GeneratorConstants.from_snapshot(ds.config.get("constants") or {})
    r = lam_residuals(ds)
    resid = r["resid"].to_numpy()
    sigma = constants.noise_sigma

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.5, wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    x = np.linspace(resid.min(), resid.max(), 300)
    pdf = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    ax.hist(resid, bins=120, density=True, color=BLUE, alpha=0.55, label="residual")
    ax.plot(x, pdf, color=ORANGE, lw=2, label=f"N(0, {sigma:.2f}) — configured noise_sigma")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_title(
        f"1 · log(lam_true / known) − mood — mean {resid.mean():.4f}, sd {resid.std():.4f} "
        f"vs configured {sigma:.2f}",
        loc="left", color=INK,
    )
    ax.set_xlabel("residual (log space)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y")

    for i, (label, key) in enumerate((
        ("by month", r["stay_date"].dt.month),
        ("by booking weekday", r["stay_date"].dt.weekday),
        ("by year", r["stay_date"].dt.year),
    )):
        ax = fig.add_subplot(gs[1, i])
        by_group = r.groupby(key)["resid"]
        means = by_group.mean()
        tol = _tol(resid.std(), by_group.size())
        ax.axhspan(-tol, tol, color=MUTED, alpha=0.18, lw=0)
        ax.bar(means.index.astype(str), means.to_numpy(), color=BLUE, width=0.7)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(f"2.{i + 1} · mean resid, {label}", loc="left", color=INK, fontsize=9.5)
        ax.set_ylim(-4 * tol, 4 * tol)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y")

    _header(
        fig, "R1 — the lam residual is exactly the configured noise draw, nothing else",
        f"{ds.root.name}: dividing every demand_factors() term out of lam_true leaves a residual "
        "flat inside its tolerance band (shaded) in every grouping — a tilt in any one panel "
        "would name the broken factor",
        top=0.88,
    )
    return _save(fig, out_path, "R1: lam residual histogram and no-leftover-structure checks.")


def fig_shape_ratio(ds: LoadedDataset, out_path: Path) -> Figure:
    """R2: actual/expected booking curve ratio across DTA — the capacity-censoring gradient."""
    plt = _style()
    dtas, actual, expected = shape_curves(ds)
    ratio = actual / expected
    corr = float(np.corrcoef(actual, expected)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    ax.plot(dtas, actual, color=BLUE, label="actual (realized bookings)")
    ax.plot(dtas, expected, color=ORANGE, ls=(0, (4, 2)), label="expected (lam · shape)")
    ax.set_xscale("log")
    ax.set_xlim(dtas.max(), 1)
    ax.set_title(f"1 · portfolio-summed curve  (corr {corr:.4f})", loc="left", color=INK)
    ax.set_xlabel("DTA (log scale)")
    ax.set_ylabel("gross bookings, summed over the portfolio")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y")

    ax = axes[1]
    ax.plot(dtas, ratio, color=BLUE)
    ax.axhline(1.0, color=AXIS, lw=1)
    ax.axvspan(240, dtas.max(), color=MUTED, alpha=0.18, lw=0)
    ax.annotate("uncensored tail\n(the R2 level check)", (dtas.max() * 0.78, 0.75),
                color=INK_2, fontsize=8, ha="center")
    ax.set_xlim(dtas.max(), 0)
    ax.set_ylim(0.5, 1.15)
    ax.set_title("2 · ratio: capacity rejection suppresses low-DTA bookings", loc="left",
                 color=INK)
    ax.set_xlabel("DTA")
    ax.set_ylabel("actual / expected")
    ax.grid(axis="y")

    _header(
        fig, "R2 — the recovered shape tracks realized bookings, with the expected censoring gap",
        f"{ds.root.name}: correlation {corr:.4f} on the raw curves; the low-DTA shortfall is "
        "curve.apply_capacity's sell-out plateau, not a shape error — it vanishes by DTA 240",
        top=0.87,
    )
    return _save(fig, out_path, "R2: actual vs expected booking curve, and their ratio.")


def fig_shock_bands(ds: LoadedDataset, out_path: Path) -> Figure:
    """R3: empirical vs theoretical shock-band correlation — the shock_corr_len signature."""
    plt = _style()
    constants = GeneratorConstants.from_snapshot(ds.config.get("constants") or {})
    empirical, theoretical = shock_band_correlations(ds)
    labels = [b.replace("shock_band_", "").replace("_", "-") for b in SHOCK_BANDS]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, mat, title in ((axes[0], empirical, "empirical"), (axes[1], theoretical, "theoretical")):
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8, color=INK_2)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=8, color=INK_2)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat.to_numpy()[i, j]:.2f}", ha="center", va="center",
                        color=INK if mat.to_numpy()[i, j] < 0.6 else SURFACE, fontsize=8.5)
        ax.set_title(f"{title} correlation", loc="left", color=INK, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)

    gap = float(np.abs((empirical - theoretical).to_numpy())[np.triu_indices(4, k=1)].mean())
    _header(
        fig, "R3 — shock_band_* correlations match the AR(1) model's exp(-Δk/corr_len) prediction",
        f"{ds.root.name}: shock_corr_len={constants.shock_corr_len:.0f}d, mean |empirical - "
        f"theoretical| = {gap:.3f} across the six band pairs",
        top=0.86,
    )
    return _save(fig, out_path, "R3: empirical vs theoretical shock-band correlation matrices.")


def _dow_table(ds: LoadedDataset) -> pd.DataFrame:
    start = pd.Timestamp(ds.config["start"])
    sd = _sd_with_latents(ds)
    rows = []
    for archetype, g in sd.groupby("archetype", sort=True):
        g = g.copy()
        g["isolated"] = _isolate(g, start, exclude="dow")
        recovered = np.exp(g.groupby(g["stay_date"].dt.weekday)["isolated"].mean())
        for wd, configured in enumerate(ARCHETYPES[archetype].dow):
            rows.append(dict(category="arrival DOW", archetype=archetype, key=str(wd),
                              configured=configured, recovered=float(recovered.get(wd, np.nan))))
    return pd.DataFrame(rows)


def _holiday_table(ds: LoadedDataset) -> pd.DataFrame:
    start = pd.Timestamp(ds.config["start"])
    sd = _sd_with_latents(ds)
    years = sorted(sd["stay_date"].dt.year.unique())
    rows = []
    for archetype, g in sd.groupby("archetype", sort=True):
        g = g.copy()
        g["isolated"] = _isolate(g, start, exclude="holiday")
        for name, configured in ARCHETYPES[archetype].hol.items():
            targets = pd.to_datetime(
                [holiday_dates(y)[name] for y in years if name in holiday_dates(y)]
            )
            mask = g["stay_date"].isin(targets)
            if mask.sum() < 20:
                continue
            recovered = float(np.exp(g.loc[mask, "isolated"].mean()))
            rows.append(dict(category="holiday", archetype=archetype, key=name,
                              configured=configured, recovered=recovered))
    return pd.DataFrame(rows)


def _monthly_table(ds: LoadedDataset) -> pd.DataFrame:
    start = pd.Timestamp(ds.config["start"])
    sd = _sd_with_latents(ds)
    rows = []
    for archetype, g in sd.groupby("archetype", sort=True):
        g = g.copy()
        g["isolated"] = _isolate(g, start, exclude="monthly")
        recovered = np.exp(g.groupby(g["stay_date"].dt.month)["isolated"].mean())
        for m, configured in enumerate(ARCHETYPES[archetype].month, start=1):
            rows.append(dict(category="month", archetype=archetype, key=str(m),
                              configured=configured, recovered=float(recovered.get(m, np.nan))))
    return pd.DataFrame(rows)


def fig_calendar_parity(ds: LoadedDataset, out_path: Path) -> Figure:
    """R4/R5/R7 (scalar parts): recovered vs configured, every archetype-keyed factor at once."""
    plt = _style()
    tables = {
        "arrival DOW": _dow_table(ds), "holiday": _holiday_table(ds), "month": _monthly_table(ds),
    }
    markers = {"arrival DOW": "o", "holiday": "s", "month": "^"}

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    lo = min(t["configured"].min() for t in tables.values()) - 0.05
    hi = max(t["configured"].max() for t in tables.values()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)

    for label, t in tables.items():
        for archetype, g in t.groupby("archetype"):
            ax.scatter(g["configured"], g["recovered"], marker=markers[label],
                       color=ARCHETYPE_COLOURS[archetype], s=32, alpha=0.85,
                       edgecolor=SURFACE, linewidth=0.4, zorder=3)

    max_gap = max(float((t["recovered"] - t["configured"]).abs().max()) for t in tables.values())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("configured"); ax.set_ylabel("recovered")
    ax.set_title(f"max |recovered - configured| = {max_gap:.3f}, across {sum(len(t) for t in tables.values())} points",
                 loc="left", color=INK, fontsize=10)
    ax.grid(True)

    from matplotlib.lines import Line2D
    shape_handles = [Line2D([0], [0], marker=markers[k], color=INK_2, lw=0, label=k)
                     for k in tables]
    arch_handles = [Line2D([0], [0], marker="o", color=c, lw=0, label=a)
                    for a, c in ARCHETYPE_COLOURS.items()]
    leg1 = ax.legend(handles=shape_handles, loc="upper left", fontsize=8, title="effect",
                      title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=arch_handles, loc="lower right", fontsize=7.5, title="archetype",
              title_fontsize=8)

    _header(
        fig, "R4/R5/R7 — every archetype-keyed calendar factor recovers on the diagonal",
        f"{ds.root.name}: DOW, holiday, and month multipliers, recovered per archetype",
        top=0.89,
    )
    return _save(fig, out_path, "R4/R5/R7: recovered-vs-configured parity plot.")


def fig_yearly_seasonality(ds: LoadedDataset, out_path: Path,
                            archetypes: tuple[str, str] = ("urban_business", "resort")) -> Figure:
    """R7 (yearly): the smooth Fourier factor, recovered pointwise per exact day-of-year."""
    plt = _style()
    start = pd.Timestamp(ds.config["start"])
    sd = _sd_with_latents(ds)

    fig, axes = plt.subplots(1, len(archetypes), figsize=(11.5, 4.2), sharey=True)
    for ax, archetype in zip(axes, archetypes, strict=True):
        A = ARCHETYPES[archetype]
        g = sd[sd["archetype"] == archetype].copy()
        g["isolated_year"] = _isolate(g, start, exclude="yearly")
        doy = g["stay_date"].dt.dayofyear
        recovered = np.exp(g.groupby(doy)["isolated_year"].mean()).reindex(range(1, 367))

        doy_grid = np.arange(1, 367)
        configured = np.exp(sum(
            A.yearly["a"][h] * np.cos(2 * np.pi * (h + 1) * doy_grid / 365.25)
            + A.yearly["b"][h] * np.sin(2 * np.pi * (h + 1) * doy_grid / 365.25)
            for h in range(2)
        ))
        ax.scatter(recovered.index, recovered.to_numpy(), color=BLUE, s=4, alpha=0.35,
                   label="recovered (daily)")
        ax.plot(doy_grid, configured, color=ORANGE, lw=2, label="configured (2-harmonic)")
        ax.set_title(archetype, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("day of year")
        ax.set_xlim(1, 366)
        ax.grid(axis="y")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("yearly factor")

    _header(
        fig, "R7 (yearly) — the smooth seasonal factor, checked pointwise per exact calendar day",
        f"{ds.root.name}: recovered per-day-of-year mean (scatter) against the deterministic "
        "2-harmonic Fourier curve the generator actually applies (line) — no refitting",
        top=0.87,
    )
    return _save(fig, out_path, "R7: recovered vs configured yearly Fourier factor.")
