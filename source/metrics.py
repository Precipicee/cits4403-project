"""Pure metric functions of arrays."""
from __future__ import annotations

import networkx as nx
import numpy as np


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total <= 0:
        return float("nan")
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)


def lorenz(x) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    p = np.arange(len(x) + 1) / len(x)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    return p, cum / cum[-1] if cum[-1] > 0 else cum


def top_share(x, q: float = 0.1) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(round(q * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def bottom_share(x, q: float = 0.5) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(round(q * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def theil(x) -> float:
    x = np.asarray(x, dtype=float)
    mu = x.mean() if len(x) else 0.0
    if mu <= 0 or (x < 0).any():
        return float("nan")
    r = x / mu
    return float(np.mean(np.where(r > 0, r * np.log(np.where(r > 0, r, 1.0)), 0.0)))


def theil_decomposition(x, groups) -> tuple[float, float, float]:
    """Total, within-group and between-group Theil index for boolean ``groups``."""
    x = np.asarray(x, dtype=float)
    total, mu, n = theil(x), x.mean(), len(x)
    if np.isnan(total):
        return (float("nan"),) * 3
    within = between = 0.0
    for g in (groups, ~groups):
        if g.sum() == 0:
            continue
        mu_g = x[g].mean()
        share = g.sum() * mu_g / (n * mu)
        if share > 0:
            within += share * theil(x[g])
            between += share * np.log(mu_g / mu)
    return total, float(within), float(between)


def rankdata(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    vals, inverse = np.unique(a, return_inverse=True)
    sums = np.bincount(inverse, ranks)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def components(adjacency: np.ndarray) -> np.ndarray:
    """Connected-component label per node by min-label propagation."""
    n = len(adjacency)
    labels = np.arange(n)
    while True:
        neighbour_min = np.where(adjacency, labels[None, :], n).min(axis=1)
        new = np.minimum(labels, neighbour_min)
        new = new[new]
        if np.array_equal(new, labels):
            return labels
        labels = new


def largest_component_share(adjacency: np.ndarray) -> float:
    return float(np.bincount(components(adjacency)).max() / len(adjacency))


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, rng=None, axis: int = 0):
    """Percentile bootstrap interval for the mean along ``axis`` (resampling that axis)."""
    values = np.asarray(values, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    n = values.shape[axis]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.nanmean(np.take(values, idx, axis=axis), axis=axis + 1)
    alpha = (1 - level) / 2
    return np.nanquantile(means, alpha, axis=0), np.nanquantile(means, 1 - alpha, axis=0)


def assortativity(graph: nx.Graph, attribute: str) -> float:
    try:
        values = set(nx.get_node_attributes(graph, attribute).values())
        if graph.number_of_edges() == 0 or len(values) < 2:
            return float("nan")
        return float(nx.attribute_assortativity_coefficient(graph, attribute))
    except (ZeroDivisionError, ValueError):
        return float("nan")


def communities(graph: nx.Graph, method: str, seed: int):
    if graph.number_of_edges() == 0:
        return [{v} for v in graph.nodes]
    if method == "louvain":
        return nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return list(nx.community.greedy_modularity_communities(graph, weight="weight"))


def modularity_and_purity(graph: nx.Graph, groups: np.ndarray, method: str, seed: int):
    parts = communities(graph, method, seed)
    if graph.number_of_edges() == 0:
        return float("nan"), float("nan"), len(parts)
    q = nx.community.modularity(graph, parts, weight="weight")
    purity = sum(max(groups[list(c)].sum(), len(c) - groups[list(c)].sum()) for c in parts) / len(groups)
    return float(q), float(purity), len(parts)
