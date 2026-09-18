"""City state held as numpy arrays, and construction of the initial city."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .population import initial_wealth, place
from .strategies import STRATEGY_KEYS
from .topology import TOPOLOGIES

NEVER = -(10**6)


@dataclass
class Streams:
    topology: np.random.Generator
    population: np.random.Generator
    dynamics: np.random.Generator
    metrics: np.random.Generator


def make_streams(root_seed: int, repeat: int) -> Streams:
    """Four independent generators per run. The seed depends only on the repeat index, so every
    cell of an experiment shares random streams for a given repeat (common random numbers)."""
    children = np.random.SeedSequence(entropy=root_seed, spawn_key=(repeat,)).spawn(4)
    return Streams(*(np.random.Generator(np.random.PCG64(c)) for c in children))


@dataclass
class CityState:
    n: int
    t: int
    strategy: np.ndarray
    giver: np.ndarray
    style: np.ndarray
    wealth: np.ndarray
    income: np.ndarray
    income_buffer: np.ndarray
    payoff_total: np.ndarray
    games_played: np.ndarray
    coop_moves: np.ndarray
    adjacency: np.ndarray
    weight: np.ndarray
    last_contact: np.ndarray
    last_game: np.ndarray
    capacity: np.ndarray
    distance: np.ndarray
    distance_dirty: bool
    last_move: np.ndarray
    prev_move: np.ndarray
    grudge: np.ndarray
    coop_ema: np.ndarray
    rating: np.ndarray
    received: np.ndarray
    weights_version: int = 0
    _weight_cache: tuple | None = None
    _kernel_cache: tuple | None = None

    def weights_changed(self) -> None:
        self.weights_version += 1

    def effective_weight(self, decay_rate: float) -> np.ndarray:
        """Tie weights decayed to the current round (read-only; cached until ties change)."""
        key = (self.t, decay_rate, self.weights_version)
        if self._weight_cache is None or self._weight_cache[0] != key:
            w = self.weight.copy() if decay_rate == 0 else self.weight * np.exp(-decay_rate * (self.t - self.last_contact))
            w.flags.writeable = False
            self._weight_cache = (key, w)
        return self._weight_cache[1]

    def kernel(self, enc) -> np.ndarray:
        """Distance-kernel weights for the current distance matrix (read-only; cached)."""
        from .encounter import KERNELS
        key = (enc.kernel, enc.d_max, enc.distance_exponent)
        if self._kernel_cache is None or self._kernel_cache[0] is not self.distance or self._kernel_cache[1] != key:
            k = KERNELS.create(enc.kernel).weights(self.distance, enc)
            k.flags.writeable = False
            self._kernel_cache = (self.distance, key, k)
        return self._kernel_cache[2]

    def strength(self, decay_rate: float) -> np.ndarray:
        return self.effective_weight(decay_rate).sum(axis=1)

    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    def free_capacity(self) -> np.ndarray:
        return self.capacity - self.degree()

    def to_networkx(self, decay_rate: float) -> nx.Graph:
        w = self.effective_weight(decay_rate)
        g = nx.Graph()
        for i in range(self.n):
            g.add_node(i, strategy=STRATEGY_KEYS[self.strategy[i]], giver=bool(self.giver[i]),
                       wealth=float(self.wealth[i]), rating=float(self.rating[i]))
        rows, cols = np.nonzero(np.triu(self.adjacency, 1))
        g.add_weighted_edges_from(zip(rows.tolist(), cols.tolist(), w[rows, cols].tolist()))
        return g


def build_city(cfg, streams: Streams) -> CityState:
    from .encounter import distance_matrix
    from .popularity import POPULARITY

    n = cfg.population.n_agents
    adjacency = TOPOLOGIES.create(cfg.topology.name).build(n, cfg.topology, streams.topology)
    degree = adjacency.sum(axis=1)
    strategy = place(adjacency, cfg.population, streams.population)
    classes = cfg.strategies.classes
    giver = np.array([classes[STRATEGY_KEYS[s]] == "giver" for s in strategy])
    wealth = initial_wealth(cfg.population, streams.population)

    style = np.zeros(n, dtype=np.int8)
    if cfg.ties.capacity == "initial_degree":
        capacity = degree.astype(np.int16)
    elif cfg.ties.capacity == "uniform":
        capacity = np.full(n, cfg.ties.capacity_k, dtype=np.int16)
    else:
        capacity = np.full(n, n - 1, dtype=np.int16)
    if cfg.population.style_mix is not None:
        broad = streams.population.random(n) < cfg.population.style_mix
        style = np.where(broad, 1, 2).astype(np.int8)
        capacity = np.where(broad, cfg.population.k_broad, cfg.population.k_narrow).astype(np.int16)

    def minus_one():
        return np.full((n, n), -1, dtype=np.int8)

    state = CityState(
        n=n, t=0, strategy=strategy, giver=giver, style=style, wealth=wealth,
        income=np.zeros(n), income_buffer=np.zeros((cfg.metrics.income_window, n)),
        payoff_total=np.zeros(n), games_played=np.zeros(n, dtype=np.int64),
        coop_moves=np.zeros(n, dtype=np.int64), adjacency=adjacency,
        weight=adjacency * cfg.ties.initial_weight, last_contact=np.zeros((n, n), dtype=np.int32),
        last_game=np.full((n, n), NEVER, dtype=np.int32), capacity=capacity,
        distance=distance_matrix(adjacency, cfg.encounter.d_max), distance_dirty=False,
        last_move=minus_one(), prev_move=minus_one(), grudge=np.zeros((n, n), dtype=bool),
        coop_ema=np.full(n, cfg.popularity.initial_cooperation), rating=np.ones(n),
        received=np.zeros(n, dtype=np.int32),
    )
    POPULARITY.create(cfg.popularity.rule).initialise(state, cfg)
    return state
