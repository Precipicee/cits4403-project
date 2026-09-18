"""Who plays whom: distance matrix, distance kernels and encounter rules."""
from __future__ import annotations

import numpy as np

from .registry import Registry

KERNELS = Registry("kernel")
ENCOUNTERS = Registry("encounter")


def distance_matrix(adjacency: np.ndarray, d_max: int) -> np.ndarray:
    """Path lengths up to ``d_max`` by boolean frontier expansion; ``d_max + 1`` beyond range."""
    n = len(adjacency)
    dist = np.full((n, n), d_max + 1, dtype=np.int16)
    np.fill_diagonal(dist, 0)
    reached = np.eye(n, dtype=bool)
    frontier = np.eye(n)
    a = adjacency.astype(float)
    for d in range(1, d_max + 1):
        new = ((frontier @ a) > 0) & ~reached
        dist[new] = d
        reached |= new
        frontier = new.astype(float)
    return dist


@KERNELS.register("power")
class PowerKernel:
    def weights(self, dist, cfg):
        k = np.zeros(dist.shape)
        cand = (dist >= 1) & (dist <= cfg.d_max)
        k[cand] = 1.0 if cfg.distance_exponent == 0 else dist[cand].astype(float) ** -cfg.distance_exponent
        return k


@KERNELS.register("rank")
class RankKernel:
    # Liben-Nowell et al.: befriending probability falls with the number of people closer.
    def weights(self, dist, cfg):
        k = np.zeros(dist.shape)
        for d in range(1, cfg.d_max + 1):
            closer = (dist < d).sum(axis=1, keepdims=True).astype(float)
            at_d = dist == d
            k[at_d] = np.broadcast_to(closer, dist.shape)[at_d] ** -cfg.distance_exponent
        return k


def scores(state, cfg, rows=None) -> np.ndarray:
    """Attractiveness ``(w_eff + eps)^beta x R^gamma x K(d)`` for candidate partners (0 elsewhere)."""
    enc = cfg.encounter
    k = state.kernel(enc)
    s = (k if rows is None else k[rows]).copy()
    if enc.weight_exponent != 0:
        w = state.effective_weight(cfg.ties.decay_rate)
        w = w if rows is None else w[rows]
        s *= (w + enc.eps) ** enc.weight_exponent
    if enc.popularity_exponent != 0:
        s *= state.rating[None, :] ** enc.popularity_exponent
    return s


def _choose(row_cumsum: np.ndarray, u: np.ndarray) -> np.ndarray:
    total = row_cumsum[:, -1]
    idx = (row_cumsum < (u * total)[:, None]).sum(axis=1)
    return np.minimum(idx, row_cumsum.shape[1] - 1)


@ENCOUNTERS.register("sampled")
class SampledEncounter:
    def pairs(self, state, cfg, rng, agents=None):
        """Requests from ``agents`` (all agents by default), capped by ``max_received``.
        Returns (pairs, idle, dropped)."""
        enc = cfg.encounter
        agents = np.arange(state.n) if agents is None else np.asarray(agents)
        s = scores(state, cfg, rows=agents)
        cum = np.cumsum(s, axis=1)
        active = cum[:, -1] > 0
        u = rng.random((len(agents), enc.initiated))
        initiators, partners = [], []
        for b in range(enc.initiated):
            chosen = _choose(cum, u[:, b])
            initiators.append(agents[active])
            partners.append(chosen[active])
        initiators, partners = np.concatenate(initiators), np.concatenate(partners)
        idle = int((~active).sum())
        if enc.max_received is None:
            np.add.at(state.received, partners, 1)
            return np.column_stack([initiators, partners]), idle, 0
        cap = enc.max_received
        row_of = {a: r for r, a in enumerate(agents)}
        accepted, dropped = [], 0
        for r in rng.permutation(len(initiators)):
            i, j = int(initiators[r]), int(partners[r])
            if state.received[j] >= cap:
                if enc.on_saturated == "drop":
                    dropped += 1
                    continue
                j = -1
                for _ in range(enc.redraw_attempts):
                    row = s[row_of[i]] * (state.received < cap)
                    total = row.sum()
                    if total <= 0:
                        break
                    j = int(min(np.searchsorted(np.cumsum(row), rng.random() * total, side="right"), state.n - 1))
                    if state.received[j] < cap:
                        break
                    j = -1
                if j < 0:
                    dropped += 1
                    continue
            state.received[j] += 1
            accepted.append((i, j))
        pairs = np.array(accepted, dtype=np.int64).reshape(-1, 2)
        return pairs, idle, dropped


@ENCOUNTERS.register("all_edges")
class AllEdgesEncounter:
    def pairs(self, state, cfg, rng, agents=None):
        rows, cols = np.nonzero(np.triu(state.adjacency, 1))
        order = rng.permutation(len(rows))
        pairs = np.column_stack([rows[order], cols[order]]).astype(np.int64)
        np.add.at(state.received, pairs[:, 1], 1)
        return pairs, 0, 0
