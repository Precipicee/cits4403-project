"""Shared figure style and encodings used across every notebook."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Colour follows the entity everywhere: one colour per class, strategy and topology.
CLASS_COLOURS = {"giver": "#2a78d6", "taker": "#eb6834"}
STRATEGY_COLOURS = {"allc": "#2a78d6", "alld": "#eb6834", "tft": "#1baf7a", "gtft": "#eda100",
                    "tf2t": "#e87ba4", "grudger": "#008300", "pavlov": "#4a3aa7", "random": "#e34948",
                    "joss": "#8a4b1c"}
STRATEGY_LABELS = {"allc": "AllC", "alld": "AllD", "tft": "TFT", "gtft": "GTFT", "tf2t": "TF2T",
                   "grudger": "Grudger", "pavlov": "Pavlov", "random": "Random", "joss": "Joss"}
TOPOLOGY_COLOURS = {"erdos_renyi": "#4a3aa7", "watts_strogatz": "#008300", "barabasi_albert": "#e34948"}
TOPOLOGY_STYLES = {"erdos_renyi": "-", "watts_strogatz": "--", "barabasi_albert": ":"}
TOPOLOGY_MARKERS = {"erdos_renyi": "o", "watts_strogatz": "s", "barabasi_albert": "^"}
TOPOLOGY_LABELS = {"erdos_renyi": "Erdős–Rényi", "watts_strogatz": "Watts–Strogatz", "barabasi_albert": "Barabási–Albert"}
STAGE_COLOURS = {"M0": MUTED, "M1": "#4a3aa7", "M1-edges": "#1baf7a"}
STAGE_STYLES = {"M0": "--", "M1": "-", "M1-edges": ":"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 5,
        "font.family": "sans-serif", "figure.dpi": 110, "figure.constrained_layout.use": True,
    })


def band(ax, x, mean, lo, hi, colour, label=None, linestyle="-", marker=None, direct_label=False):
    """Mean line with a shaded 95% interval; optional label at the line's end."""
    x, mean = np.asarray(x), np.asarray(mean)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=colour, label=label, linestyle=linestyle, marker=marker,
            markersize=4 if marker else None)
    if direct_label and label is not None and len(x):
        ax.annotate(label, (x[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)


def over_time(ax, means, metric, groups, colours, group_col="scenario", ylabel=None,
              direct_label=False, linestyles=None):
    """Mean over seeds against round, one banded line per group.

    Consumes :func:`utilities.runs.series_means`, so the band is one standard error of the
    mean either side: it shows where the scenarios are actually distinguishable, not the
    spread of individual cities.
    """
    for g in groups:
        rows = means[means[group_col] == g].sort_values("round")
        m, se = rows[metric].to_numpy(float), rows[f"{metric}_se"].to_numpy(float)
        band(ax, rows["round"], m, m - se, m + se, colours[g], label=g,
             linestyle=(linestyles or {}).get(g, "-"), direct_label=direct_label)
    ax.set_xlabel("week")
    ax.set_ylabel(ylabel or metric.replace("_", " "))


def delta_bars(ax, table, xlabel=None, label_col="step", value_col="mean", se_col="se"):
    """Paired step-to-step changes as bars with standard-error whiskers.

    Bars are coloured by sign rather than by entity: this figure is about direction, and each
    row is a different comparison rather than a different thing.
    """
    y = np.arange(len(table))
    values = table[value_col].to_numpy(float)
    ax.barh(y, values, xerr=table[se_col].to_numpy(float), height=0.62,
            color=[CLASS_COLOURS["giver"] if v >= 0 else CLASS_COLOURS["taker"] for v in values],
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y, list(table[label_col]))
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(xlabel or value_col)
    ax.grid(axis="y", visible=False)


def lorenz_curves(ax, curves: dict, colours=None, ginis: dict | None = None):
    """Lorenz curves keyed by label, against the line of equality."""
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for (label, (p, l)), colour in zip(curves.items(), colours or SERIES):
        text = f"{label} (Gini {ginis[label]:.2f})" if ginis else label
        ax.plot(p, l, color=colour, linewidth=1.6, label=text)
    ax.set(xlabel="share of citizens, poorest first", ylabel="share of wealth")
    ax.set_aspect("equal")


def cell_series(cells, metric, x_col, **filters):
    """Rows of an aggregated ``cells`` table for one metric, sorted by ``x_col``."""
    q = cells[cells["metric"] == metric]
    for k, v in filters.items():
        q = q[q[k] == v]
    return q.sort_values(x_col)


def by_giver_fraction(ax, cells, metric, groups=None, group_col=None, ylabel=None, direct_label=True):
    """Metric against giver fraction, one banded line per group (e.g. topology)."""
    groups = groups or [None]
    for g in groups:
        rows = cell_series(cells, metric, "population.giver_fraction", **({group_col: g} if g else {}))
        colour = TOPOLOGY_COLOURS.get(g, SERIES[0]) if g else SERIES[0]
        band(ax, rows["population.giver_fraction"], rows["mean"], rows["ci_low"], rows["ci_high"], colour,
             label=TOPOLOGY_LABELS.get(g, g), linestyle=TOPOLOGY_STYLES.get(g, "-"),
             marker=TOPOLOGY_MARKERS.get(g), direct_label=direct_label and g is not None)
    ax.set_xlabel("giver fraction")
    ax.set_ylabel(ylabel or metric.replace("_", " "))


def headline_figure(cells, topologies, decay_col="ties.decay_rate", decay=None):
    """Mean and median wealth above, Gini below, against giver fraction, one line per topology."""
    if decay is not None:
        cells = cells[cells[decay_col] == decay]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    by_giver_fraction(top, cells, "mean_wealth", topologies, "topology.name", ylabel="wealth per agent")
    for t in topologies:
        rows = cell_series(cells, "median_wealth", "population.giver_fraction", **{"topology.name": t})
        top.plot(rows["population.giver_fraction"], rows["mean"], color=TOPOLOGY_COLOURS[t], linewidth=1,
                 linestyle=TOPOLOGY_STYLES[t], alpha=0.6)
    top.set_title("Mean (thick) and median (thin) wealth")
    by_giver_fraction(bottom, cells, "gini", topologies, "topology.name", ylabel="Gini coefficient")
    bottom.set_title("Inequality of wealth")
    top.legend(loc="upper left")
    return fig


def lorenz_overlay(ax, curves: dict, cmap="Blues"):
    """Lorenz curves keyed by a numeric label (e.g. giver fraction), light to dark by value."""
    keys = sorted(curves)
    colours = mpl.colormaps[cmap](np.linspace(0.35, 0.95, len(keys)))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for colour, k in zip(colours, keys):
        p, l = curves[k]
        ax.plot(p, l, color=colour, linewidth=1.5, label=f"{k:g}")
    ax.set_xlabel("share of agents (poorest first)")
    ax.set_ylabel("share of wealth")
    ax.set_aspect("equal")


def ramp(values, cmap="Blues"):
    """One colour per ordered value, light to dark."""
    values = sorted(values)
    colours = mpl.colormaps[cmap](np.linspace(0.35, 0.95, len(values)))
    return dict(zip(values, colours))
