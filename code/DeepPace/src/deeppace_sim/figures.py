"""Plots of the generator's own functions.

Spec: docs/simulation-spec.md. Rendered by `scripts/plot_functions.py` into `reports/figures/`.

**None of these needs a dataset.** That is a property of the top-down design rather than a
convenience: `demand_factors` and `pace_shape` are ordinary functions returning arrays, so the
model can be inspected directly. Under the bottom-up predecessor there was no booking-pace
function to plot — the curve existed only after simulating a stream of reservations, so the only
way to see what the generator believed was to generate data and measure it back.

Three figures, one per stage of `nu = lam · shape · shock`:

    fig_demand_function   the multiplicative factor stack that sets a date's level
    fig_pace_function     the booking curve: components, archetype mixtures, compression
    fig_shock_field       the deviation field, and the composition into an on-books curve

Matplotlib is imported lazily inside each function: generating data must not require a plotting
stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from deeppace_sim.config import ARCHETYPES, COMPONENTS, GeneratorConstants
from deeppace_sim.demand import demand_factors
from deeppace_sim.pace import (
    booking_calendar_factor,
    compression_shift,
    lognormal_pmf,
    mixture_weights,
    pace_shape,
    pace_shock,
)

# Validated default palette (dataviz reference instance), light surface. Line charts use the
# adjacent pairlist, on which all eight slots clear the CVD and normal-vision gates; every series
# is also directly labelled, so identity never rests on colour alone.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
)


@dataclass(frozen=True)
class Figure:
    path: Path
    caption: str


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2.0,
    })
    return plt


def _header(fig, title: str, subtitle: str, top: float = 0.87) -> None:
    fig.text(0.006, 0.975, title, ha="left", va="top", color=INK, fontsize=12)
    fig.text(0.006, 0.925, subtitle, ha="left", va="top", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, top))


def _save(fig, out_path: Path, caption: str) -> Figure:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return Figure(out_path, caption)


# --------------------------------------------------------------------------------------


def fig_demand_function(
    out_path: Path,
    archetypes: tuple[str, str] = ("urban_business", "resort"),
    year: int = 2024,
    start: date = date(2023, 1, 1),
) -> Figure:
    """The multiplicative factor stack that sets a stay date's demand level.

    Two archetypes side by side, because the holiday factor **inverts between them**: Christmas is
    0.28x for an urban business hotel and 1.50x for a resort. A model that learns one global
    holiday effect is badly wrong on half the portfolio, which is the point of showing it.
    """
    plt = _style()
    days = [date(year, 1, 1) + timedelta(days=i) for i in range(365)]
    doy = np.arange(1, 366)
    named = ("yearly", "monthly", "dow", "holiday", "trend")

    fig, axes = plt.subplots(
        2, len(archetypes), figsize=(11.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
    )
    # the factor rows share a y-scale so the holiday inversion is directly comparable; the
    # product rows do not, because the archetypes genuinely reach different levels
    axes[0, 1].sharey(axes[0, 0])
    for col, archetype in enumerate(archetypes):
        A = ARCHETYPES[archetype]
        factors = [demand_factors(A, 1.0, d, start) for d in days]

        ax = axes[0, col]
        for i, (colour, name) in enumerate(zip(SERIES, named, strict=True)):
            series = np.array([f[name] for f in factors])
            # DOW swings every three days and would otherwise hide the factors that carry the
            # story, so it sits behind them, thinner and faded
            weekly = name == "dow"
            ax.plot(doy, series, color=colour, lw=0.7 if weekly else 1.8,
                    alpha=0.45 if weekly else 1.0, zorder=1 if weekly else 3)
            at = 300 - 60 * i if not weekly else 40
            ax.annotate(name, (doy[at], series[at]), xytext=(4, 5),
                        textcoords="offset points", color=colour, fontsize=8.5, zorder=4,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2, alpha=0.8))
        ax.axhline(1.0, color=AXIS, lw=1)
        ax.set_title(f"{archetype} — factors", loc="left", color=INK)
        ax.grid(axis="y")
        if col == 0:
            ax.set_ylabel("multiplier")

        ax = axes[1, col]
        level = np.array([
            np.prod([v for k, v in f.items() if k != "level"]) for f in factors
        ])
        ax.fill_between(doy, 0, level, color=BLUE, alpha=0.16, lw=0)
        ax.plot(doy, level, color=BLUE)
        ax.axhline(1.0, color=AXIS, lw=1)
        ax.set_title("their product — demand relative to the property's base", loc="left",
                     color=INK)
        ax.set_xlabel(f"day of {year}")
        ax.set_xlim(1, 365)
        ax.grid(axis="y")
        if col == 0:
            ax.set_ylabel("× base demand")

    _header(fig, "The demand function: a product of independent calendar factors",
            "holiday multipliers invert by archetype — the same date is a collapse for one and a "
            "peak for the other", top=0.90)
    return _save(fig, out_path, "The demand factor stack, two archetypes.")


def _booked_by(shape: np.ndarray) -> np.ndarray:
    """Share already on the books at DTA k = the share of bookings made at DTA >= k.

    A reverse cumulative sum, not a forward one. `pace_shape` is indexed by days *to* arrival, so
    a forward cumsum gives the share still to come — the complement, and an easy plot to draw
    upside down.
    """
    return np.cumsum(shape[::-1])[::-1]


def _label_on_curve(ax, k, cum, y, text, colour) -> None:
    """Put a label where the curve crosses `y`, so several curves' labels do not stack up."""
    x = float(np.interp(y, cum[::-1], k[::-1]))
    ax.annotate(text, (x, y), xytext=(7, -3), textcoords="offset points",
                color=colour, fontsize=8)


def fig_pace_function(out_path: Path, k_max: int = 365) -> Figure:
    """The booking-pace function: what it is made of, and what moves it."""
    plt = _style()
    k = np.arange(k_max + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

    # --- the three lead-time components ---
    ax = axes[0]
    for colour, (name, cp) in zip(SERIES, COMPONENTS.items(), strict=False):
        pmf = lognormal_pmf(k + 0.5, cp.mu, cp.sigma)
        pmf = pmf / pmf.sum()
        ax.plot(k, pmf, color=colour)
        median = float(np.exp(cp.mu))
        ax.annotate(f"{name}\nmedian {median:.0f} d", (median, pmf[int(median)]),
                    xytext=(6, 4), textcoords="offset points", color=colour, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(k_max, 1)
    ax.set_title("1 · lead-time components", loc="left", color=INK)
    ax.set_xlabel("DTA (log scale)")
    ax.set_ylabel("share of bookings per day")
    ax.grid(axis="y")

    # --- what moves the mixture: regime, not archetype ---
    ax = axes[1]
    regimes = (
        ("weekday", date(2024, 6, 5), 0.70, False),
        ("weekend", date(2024, 6, 8), 0.70, False),
        ("weekend, busy", date(2024, 6, 8), 1.10, False),
        ("weekday, event", date(2024, 6, 5), 1.10, True),
    )
    for i, (colour, (label, d, occ, ev)) in enumerate(zip(SERIES, regimes, strict=False)):
        cum = _booked_by(pace_shape(k, mixture_weights(d, occ, ev), occ, 0.35))
        ax.plot(k, cum, color=colour)
        half = float(np.interp(0.5, cum[::-1], k[::-1]))
        _label_on_curve(ax, k, cum, 0.26 + 0.15 * i, f"{label}  {half:.0f} d", colour)
    ax.axhline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(k_max, 0)
    ax.set_ylim(0, 1)
    ax.set_title("2 · mixtures by regime", loc="left", color=INK)
    ax.set_xlabel("DTA")
    ax.set_ylabel("cumulative share booked")
    ax.grid(axis="y")

    # --- compression ---
    ax = axes[2]
    occs = (0.40, 0.70, 1.10)
    for i, (colour, occ) in enumerate(zip(SERIES, occs, strict=False)):
        weights = mixture_weights(date(2024, 6, 5), occ, False)
        cum = _booked_by(pace_shape(k, weights, occ, 0.35))
        ax.plot(k, cum, color=colour)
        half = float(np.interp(0.5, cum[::-1], k[::-1]))
        _label_on_curve(
            ax, k, cum, 0.30 + 0.18 * i,
            f"expected occ {occ:.2f} → ×{compression_shift(occ):.2f}, half booked {half:.0f} d",
            colour,
        )
    ax.axhline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(k_max, 0)
    ax.set_ylim(0, 1)
    ax.set_title("3 · compression", loc="left", color=INK)
    ax.set_xlabel("DTA")
    ax.grid(axis="y")

    _header(fig, "The booking-pace function: a lead-time mixture, shifted by expected fullness",
            "archetype does not enter — the mixture moves on regime alone (weekend, event, "
            "expected fullness); medians calibrated to a 69-day median lead", top=0.86)
    return _save(fig, out_path, "The pace function: components, mixtures, compression.")


def fig_shock_field(out_path: Path, k_max: int = 365, seed: int = 7) -> Figure:
    """The deviation field, and how a curve is composed from it.

    The middle panel is the knob that matters. `shock_corr_len` decides over how many days of DTA
    a date's deviation persists, and therefore how much of it is **visible early**: at a long
    correlation length a date running hot at DTA 200 is still running hot at DTA 50 and will
    finish high, so the partial curve carries information the level does not.
    """
    plt = _style()
    c = GeneratorConstants()
    k = np.arange(k_max + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

    # --- sample fields at three correlation lengths ---
    ax = axes[0]
    for i, (colour, corr) in enumerate(zip(SERIES, (20.0, 240.0, 2000.0), strict=False)):
        shock = pace_shock(np.random.default_rng(seed), k, c.shock_sigma, corr)
        ax.plot(k, shock, color=colour, lw=1.4 if corr < 100 else 1.8,
                alpha=0.6 if corr < 100 else 1.0)
        at = int(k_max * (0.82 - 0.26 * i))
        ax.annotate(f"corr_len {corr:.0f} d", (k[at], shock[at]), xytext=(0, 10),
                    textcoords="offset points", ha="center", color=colour, fontsize=8.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2, alpha=0.85))
    ax.axhline(1.0, color=AXIS, lw=1)
    ax.set_xlim(k_max, 0)
    ax.set_title("1 · the shock field, one seed", loc="left", color=INK)
    ax.set_xlabel("DTA")
    ax.set_ylabel("× the systematic shape")
    ax.grid(axis="y")

    # --- what that does to the realised curve ---
    ax = axes[1]
    w = mixture_weights(date(2024, 6, 5), 0.70, False)
    shape = pace_shape(k, w, 0.70, c.kappa)
    base = _booked_by(shape)
    ax.plot(k, base, color=MUTED, lw=1.6, ls=(0, (4, 3)))
    ax.annotate("no shock", (float(np.interp(0.62, base[::-1], k[::-1])), 0.62),
                xytext=(7, -3), textcoords="offset points", color=MUTED, fontsize=8.5)
    rng = np.random.default_rng(seed)
    for colour in SERIES[:4]:
        nu = shape * pace_shock(rng, k, c.shock_sigma, c.shock_corr_len)
        ax.plot(k, _booked_by(nu), color=colour, lw=1.4, alpha=0.9)
    ax.set_xlim(k_max, 0)
    ax.set_title("2 · four dates with the same lam", loc="left", color=INK)
    ax.set_xlabel("DTA")
    ax.set_ylabel("on books, × lam")
    ax.grid(axis="y")

    # --- composition for one date ---
    ax = axes[2]
    shock = pace_shock(np.random.default_rng(seed + 1), k, c.shock_sigma, c.shock_corr_len)
    promo = {date(2024, 6, 5) - timedelta(days=40): 2.2}
    calendar = booking_calendar_factor(date(2024, 6, 5), k, promo, c.book_dow)
    shaped = pace_shape(k, w, 0.70, c.kappa, calendar)
    lam = 150.0
    systematic = lam * shaped
    intensity = systematic * shock
    ax.plot(k, systematic, color=MUTED, lw=1.2)
    ax.plot(k, intensity, color=BLUE)
    ax.annotate("shape × lam", (k[95], systematic[95]), xytext=(0, -20),
                textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
    ax.annotate("× shock = intensity", (k[75], intensity[75]), xytext=(0, 16),
                textcoords="offset points", ha="center", color=BLUE, fontsize=8.5)
    ax.annotate("promo on one booking date", (40, intensity[40]),
                xytext=(-14, 46), textcoords="offset points", ha="center", color=ORANGE,
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.set_xlim(120, 0)
    ax.set_title("3 · composition, one stay date", loc="left", color=INK)
    ax.set_xlabel("DTA")
    ax.set_ylabel("rooms booked per day")
    ax.grid(axis="y")

    _header(fig, "The shock field: how far a date deviates, and how early that shows",
            f"shock_sigma {c.shock_sigma} sets the size, shock_corr_len {c.shock_corr_len:.0f} d "
            "sets how much of it a partial curve reveals", top=0.86)
    return _save(fig, out_path, "The shock field and the composition into an intensity.")
