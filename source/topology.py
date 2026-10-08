"""Initial city graphs (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, complete)."""
from __future__ import annotations

import networkx as nx
import numpy as np

from .registry import Registry

TOPOLOGIES = Registry("topology")
MAX_ATTEMPTS = 100


def _adjacency(graph: nx.Graph, n: int) -> np.ndarray:
    return nx.to_numpy_array(graph, nodelist=range(n), dtype=bool)


@TOPOLOGIES.register("complete")
class Complete:
    """Every agent tied to every other (the well-mixed city, S1)."""

    def build(self, n, cfg, rng):
        """Boolean adjacency matrix of the complete graph on ``n`` nodes."""
        a = np.ones((n, n), dtype=bool)
        np.fill_diagonal(a, False)
        return a


class _Random:
    """Shared machinery for random generators: redraw until the graph is connected, up to
    ``MAX_ATTEMPTS`` times, so no agent starts cut off from the city."""

    def generate(self, n, cfg, seed) -> nx.Graph:
        raise NotImplementedError

    def build(self, n, cfg, rng):
        """Boolean adjacency matrix of the first connected graph drawn; raises ``ConfigError``
        if none is found."""
        for _ in range(MAX_ATTEMPTS):
            graph = self.generate(n, cfg, int(rng.integers(2**32)))
            if nx.is_connected(graph):
                return _adjacency(graph, n)
        from .config import ConfigError
        raise ConfigError(f"no connected {self.key} graph after {MAX_ATTEMPTS} attempts")


@TOPOLOGIES.register("erdos_renyi")
class ErdosRenyi(_Random):
    """Erdős–Rényi random graph, with each tie present with probability ``k / (n - 1)``."""

    def generate(self, n, cfg, seed):
        return nx.gnp_random_graph(n, cfg.mean_degree / (n - 1), seed=seed)


@TOPOLOGIES.register("watts_strogatz")
class WattsStrogatz(_Random):
    """Watts–Strogatz small-world graph: a ring lattice of degree ``k`` with each tie rewired
    with probability ``ws_rewire_prob``."""

    def generate(self, n, cfg, seed):
        return nx.watts_strogatz_graph(n, cfg.mean_degree, cfg.ws_rewire_prob, seed=seed)


@TOPOLOGIES.register("barabasi_albert")
class BarabasiAlbert(_Random):
    """Barabási–Albert scale-free graph, each new node attaching ``k / 2`` ties preferentially
    to well-connected nodes, so the mean degree is close to ``k``."""

    def generate(self, n, cfg, seed):
        return nx.barabasi_albert_graph(n, cfg.mean_degree // 2, seed=seed)
