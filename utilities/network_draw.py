"""Network snapshots: node colour by strategy, size by wealth, edge width by tie strength."""
from __future__ import annotations

import networkx as nx
import numpy as np

from source.strategies import STRATEGY_KEYS

from .plotting import AXIS, CLASS_COLOURS, INK, STRATEGY_COLOURS, STRATEGY_LABELS


def layout(adjacency: np.ndarray, seed: int = 0) -> dict:
    g = nx.from_numpy_array(adjacency.astype(int))
    return nx.spring_layout(g, seed=seed, k=1.2 / np.sqrt(len(adjacency)))


def drifting_layouts(frames: list, seed: int = 0, iterations: int = 15) -> list[dict]:
    """One layout per snapshot, each started from the one before, so nodes move a little per
    frame rather than jumping. Agents with ties are laid out inside the unit circle; an agent
    with no ties at all is placed on a fixed outer ring, so exclusion reads as a node leaving
    the city rather than as the whole drawing shrinking around it."""
    n = len(frames[0]["adjacency"])
    k = 1.2 / np.sqrt(n)
    angle = 2 * np.pi * np.arange(n) / n
    ring = {i: np.array([1.15 * np.cos(angle[i]), 1.15 * np.sin(angle[i])]) for i in range(n)}
    pos = layout(frames[0]["adjacency"], seed=seed)
    out = []
    for frame in frames:
        adjacency = frame["adjacency"]
        tied = np.flatnonzero(adjacency.sum(axis=1) > 0)
        g = nx.from_numpy_array(adjacency[np.ix_(tied, tied)].astype(int))
        start = {a: pos[int(node)] for a, node in enumerate(tied)}
        inner = nx.spring_layout(g, pos=start, iterations=iterations, seed=seed, k=k) if len(tied) else {}
        xy = np.array([inner[a] for a in range(len(tied))]) if len(tied) else np.zeros((0, 2))
        if len(xy):
            xy = xy - xy.mean(axis=0)
            xy = xy / max(np.linalg.norm(xy, axis=1).max(), 1e-9)
        pos = dict(ring)
        pos.update({int(node): xy[a] for a, node in enumerate(tied)})
        out.append(pos)
    return out


def draw_snapshot(ax, snap: dict, pos: dict, title: str = "", max_size: float = 120,
                  since: np.ndarray | None = None, wealth_scale: tuple | None = None,
                  weight_scale: float | None = None, by_class: bool = False):
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
    colours = ([CLASS_COLOURS["giver" if g else "taker"] for g in snap["giver"]] if by_class
               else [STRATEGY_COLOURS[STRATEGY_KEYS[s]] for s in snap["strategy"]])
    xy = np.array([pos[i] for i in range(len(wealth))])
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colours, edgecolors="#fcfcfb", linewidths=0.8, zorder=2)
    ax.set_title(title)
    ax.set_axis_off()


def strategy_legend(ax, keys=STRATEGY_KEYS):
    handles = [ax.scatter([], [], color=STRATEGY_COLOURS[k], s=30, label=STRATEGY_LABELS[k]) for k in keys]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))


def class_legend(ax):
    handles = [ax.scatter([], [], color=CLASS_COLOURS[c], s=30, label=c) for c in ("giver", "taker")]
    ax.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))


def animate(cities: dict, path, fps: int = 6, seed: int = 3, max_size: float = 120, hold: int = 6):
    """Write a GIF of several cities side by side, one frame per snapshot, coloured by class.

    ``cities`` maps a panel title to that city's snapshots, taken at the same weeks. Each city
    gets its own drifting layout. Wealth and tie-strength scales are fixed across every panel and
    frame, so a node that stops growing, or a tie that thins, is a real change. The last frame is
    shown ``hold`` times over, so the end state stays on screen before the loop restarts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    titles = list(cities)
    n_frames = min(len(frames) for frames in cities.values())
    every = [frames[:n_frames] for frames in cities.values()]
    layouts = [drifting_layouts(frames, seed=seed) for frames in every]
    wealth_scale = (0.0, max(f["wealth"].max() for frames in every for f in frames))
    weight_scale = max(f["weight"].max() for frames in every for f in frames)
    fig, axes = plt.subplots(1, len(titles), figsize=(5.2 * len(titles), 5.6))
    axes = np.atleast_1d(axes)

    def draw(k):
        for ax, title, frames, pos in zip(axes, titles, every, layouts):
            ax.clear()
            snap = frames[k]
            draw_snapshot(ax, snap, pos[k], title=f"{title}, week {snap['round']}", max_size=max_size,
                          wealth_scale=wealth_scale, weight_scale=weight_scale, by_class=True)
        class_legend(axes[0])
        return []

    anim = FuncAnimation(fig, draw, frames=list(range(n_frames)) + [n_frames - 1] * hold, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path
