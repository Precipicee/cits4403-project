"""Network snapshots: node colour by strategy, size by wealth, edge width by tie strength."""
from __future__ import annotations

import networkx as nx
import numpy as np

from source.strategies import STRATEGY_KEYS

from .plotting import AXIS, STRATEGY_COLOURS, STRATEGY_LABELS


def layout(adjacency: np.ndarray, seed: int = 0) -> dict:
    g = nx.from_numpy_array(adjacency.astype(int))
    return nx.spring_layout(g, seed=seed, k=1.2 / np.sqrt(len(adjacency)))


def draw_snapshot(ax, snap: dict, pos: dict, title: str = "", max_size: float = 120):
    adjacency, weight = snap["adjacency"], snap["weight"]
    rows, cols = np.nonzero(np.triu(adjacency, 1))
    w = weight[rows, cols]
    widths = 0.3 + 2.2 * (w / w.max()) if len(w) and w.max() > 0 else []
    for (i, j), lw in zip(zip(rows, cols), widths):
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=AXIS, linewidth=lw, alpha=0.6, zorder=1)
    wealth = snap["wealth"]
    sizes = 8 + max_size * (wealth - wealth.min()) / (np.ptp(wealth) or 1)
    colours = [STRATEGY_COLOURS[STRATEGY_KEYS[s]] for s in snap["strategy"]]
    xy = np.array([pos[i] for i in range(len(wealth))])
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colours, edgecolors="#fcfcfb", linewidths=0.8, zorder=2)
    ax.set_title(title)
    ax.set_axis_off()


def strategy_legend(ax, keys=STRATEGY_KEYS):
    handles = [ax.scatter([], [], color=STRATEGY_COLOURS[k], s=30, label=STRATEGY_LABELS[k]) for k in keys]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
