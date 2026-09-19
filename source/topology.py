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
    def build(self, n, cfg, rng):
        a = np.ones((n, n), dtype=bool)
        np.fill_diagonal(a, False)
        return a


class _Random:
    def generate(self, n, cfg, seed) -> nx.Graph:
        raise NotImplementedError

    def build(self, n, cfg, rng):
        for _ in range(MAX_ATTEMPTS):
            graph = self.generate(n, cfg, int(rng.integers(2**32)))
            if nx.is_connected(graph):
                return _adjacency(graph, n)
        from .config import ConfigError
        raise ConfigError(f"no connected {self.key} graph after {MAX_ATTEMPTS} attempts")


@TOPOLOGIES.register("erdos_renyi")
class ErdosRenyi(_Random):
    def generate(self, n, cfg, seed):
        return nx.gnp_random_graph(n, cfg.mean_degree / (n - 1), seed=seed)


@TOPOLOGIES.register("watts_strogatz")
class WattsStrogatz(_Random):
    def generate(self, n, cfg, seed):
        return nx.watts_strogatz_graph(n, cfg.mean_degree, cfg.ws_rewire_prob, seed=seed)


@TOPOLOGIES.register("barabasi_albert")
class BarabasiAlbert(_Random):
    def generate(self, n, cfg, seed):
        return nx.barabasi_albert_graph(n, cfg.mean_degree // 2, seed=seed)
