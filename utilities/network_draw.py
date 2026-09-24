"""Network snapshots: node colour by strategy, size by wealth, edge width by tie strength."""
from __future__ import annotations

import networkx as nx
import numpy as np

from source.strategies import STRATEGY_KEYS

from .plotting import AXIS, INK, STRATEGY_COLOURS, STRATEGY_LABELS


def layout(adjacency: np.ndarray, seed: int = 0) -> dict:
    g = nx.from_numpy_array(adjacency.astype(int))
    return nx.spring_layout(g, seed=seed, k=1.2 / np.sqrt(len(adjacency)))


def draw_snapshot(ax, snap: dict, pos: dict, title: str = "", max_size: float = 120,
                  since: np.ndarray | None = None, wealth_scale: tuple | None = None,
                  weight_scale: float | None = None):
    """Draw one city. ``since`` marks ties absent from that earlier adjacency, so churn is
    visible; ``wealth_scale`` and ``weight_scale`` fix the encodings across a row of frames,
    without which every panel normalises to its own maximum and a changing city looks static.
    """
    adjacency, weight = snap["adjacency"], snap["weight"]
    rows, cols = np.nonzero(np.triu(adjacency, 1))
    w = weight[rows, cols]
    top = weight_scale or (w.max() if len(w) else 0)
    widths = 0.3 + 2.2 * (w / top) if top > 0 else []
    fresh = ~since[rows, cols] if since is not None else np.zeros(len(rows), dtype=bool)
    for (i, j), lw, new_tie in zip(zip(rows, cols), widths, fresh):
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color=INK if new_tie else AXIS, linewidth=lw,
                alpha=0.75 if new_tie else 0.5, zorder=1)
    wealth = snap["wealth"]
    lo, hi = wealth_scale or (wealth.min(), wealth.max())
    sizes = 8 + max_size * np.clip((wealth - lo) / ((hi - lo) or 1), 0, 1)
    colours = [STRATEGY_COLOURS[STRATEGY_KEYS[s]] for s in snap["strategy"]]
    xy = np.array([pos[i] for i in range(len(wealth))])
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colours, edgecolors="#fcfcfb", linewidths=0.8, zorder=2)
    ax.set_title(title)
    ax.set_axis_off()


def strategy_legend(ax, keys=STRATEGY_KEYS):
    handles = [ax.scatter([], [], color=STRATEGY_COLOURS[k], s=30, label=STRATEGY_LABELS[k]) for k in keys]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
