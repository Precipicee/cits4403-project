"""Strategy counts, placement of strategies on the graph, and initial wealth."""
from __future__ import annotations

from collections import deque

import numpy as np

from .strategies import STRATEGY_ID, STRATEGY_KEYS


def largest_remainder(total: int, weights: dict) -> dict:
    """Integer counts proportional to ``weights`` summing to ``total``; ties by registry order."""
    keys = [k for k in STRATEGY_KEYS if weights.get(k, 0) > 0]
    w = np.array([weights[k] for k in keys], dtype=float)
    exact = total * w / w.sum()
    counts = np.floor(exact).astype(int)
    remainder = exact - counts
    for idx in sorted(range(len(keys)), key=lambda x: (-remainder[x], x))[: total - counts.sum()]:
        counts[idx] += 1
    return dict(zip(keys, counts.tolist()))


def strategy_counts(pop) -> tuple[dict, dict]:
    """Number of agents of each strategy, as (givers, takers). The giver count is the giver
    fraction times the population, rounded half up; each class is then split by its composition
    weights with ``largest_remainder``."""
    n_givers = int(np.floor(pop.giver_fraction * pop.n_agents + 0.5))
    return (largest_remainder(n_givers, pop.giver_composition),
            largest_remainder(pop.n_agents - n_givers, pop.taker_composition))


def _ids(counts: dict) -> np.ndarray:
    return np.concatenate([np.full(c, STRATEGY_ID[k], dtype=np.int8) for k, c in counts.items()]
                          or [np.zeros(0, dtype=np.int8)])


def place(adjacency: np.ndarray, pop, rng) -> np.ndarray:
    """Strategy id for every node. ``"random"`` placement shuffles strategies over the nodes;
    ``"clustered"`` puts all givers in one breadth-first block grown from a random node."""
    giver_counts, taker_counts = strategy_counts(pop)
    givers, takers = _ids(giver_counts), _ids(taker_counts)
    n = pop.n_agents
    if pop.placement == "random":
        ids = np.concatenate([givers, takers])
        return ids[rng.permutation(n)]
    # clustered: givers occupy a breadth-first block grown from a random root
    root = int(rng.integers(n))
    seen, queue, block = {root}, deque([root]), []
    while queue and len(block) < len(givers):
        v = queue.popleft()
        block.append(v)
        for u in np.flatnonzero(adjacency[v]):
            if u not in seen:
                seen.add(u)
                queue.append(u)
    strategy = np.empty(n, dtype=np.int8)
    block = np.array(block, dtype=int)
    rest = np.setdiff1d(np.arange(n), block)
    strategy[block] = givers[rng.permutation(len(givers))]
    strategy[rest] = takers[rng.permutation(len(takers))]
    return strategy


def initial_wealth(pop, rng) -> np.ndarray:
    """Starting wealth. ``"zero"`` (the default) gives every agent 0; otherwise a random half of
    the agents start richer, so the starting Gini equals ``endowment_gini`` (exactly when the
    number of agents is even)."""
    n = pop.n_agents
    if pop.initial_wealth == "zero":
        return np.zeros(n)
    # Two equal groups at m(1 + 2G) and m(1 - 2G) have Gini exactly G.
    wealth = np.full(n, pop.endowment_mean * (1 - 2 * pop.endowment_gini))
    high = rng.permutation(n)[: n // 2]
    wealth[high] = pop.endowment_mean * (1 + 2 * pop.endowment_gini)
    return wealth
