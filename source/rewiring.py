"""Refilling freed tie capacity."""
from __future__ import annotations

import numpy as np

from .registry import Registry

REWIRING = Registry("rewiring rule")


@REWIRING.register("none")
class NoRewiring:
    def refill(self, state, cfg, rng) -> int:
        return 0


@REWIRING.register("local_refill")
class LocalRefill:
    """Agents with spare capacity draw new partners among non-neighbours at distance 2..d_max,
    in proportion to popularity and the distance kernel; most new ties are friends of friends."""

    def refill(self, state, cfg, rng) -> int:
        free = state.free_capacity().astype(int)
        agents = np.flatnonzero(free > 0)
        if not len(agents):
            return 0
        enc = cfg.encounter
        kernel = state.kernel(enc)
        status = state.rating ** enc.popularity_exponent
        from .encounter import refused
        closed = refused(state, cfg)
        created = 0
        for i in rng.permutation(agents):
            while free[i] > 0:
                open_ = (free > 0) & ~state.adjacency[i]
                open_[i] = False
                if closed is not None:
                    open_ &= ~closed[i]
                row = kernel[i] * status * open_ * (state.distance[i] >= 2)
                total = row.sum()
                if total <= 0:
                    if cfg.rewiring.fallback == "none" or not open_.any():
                        break
                    row, total = open_.astype(float), float(open_.sum())
                j = int(min(np.searchsorted(np.cumsum(row), rng.random() * total, side="right"), state.n - 1))
                state.adjacency[i, j] = state.adjacency[j, i] = True
                state.weight[i, j] = state.weight[j, i] = cfg.ties.initial_weight
                state.last_contact[i, j] = state.last_contact[j, i] = state.t
                free[i] -= 1
                free[j] -= 1
                created += 1
        if created:
            state.distance_dirty = True
            state.weights_changed()
        return created
