"""Tie reinforcement, formation on contact, time decay and severance."""
from __future__ import annotations

import numpy as np

from .registry import Registry

TIE_RULES = Registry("severance rule")


def reinforce(state, record, cfg) -> int:
    """Add ``delta`` per game to existing ties; form a tie on contact when both agents have
    spare capacity. Returns the number of ties formed."""
    if record.n_games == 0:
        return 0
    t, ti = state.t, cfg.ties
    i, j = record.initiator, record.partner
    state.last_game[i, j] = t
    state.last_game[j, i] = t
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    tied = state.adjacency[lo, hi]
    formed = 0
    if tied.any():
        key = lo[tied].astype(np.int64) * state.n + hi[tied]
        uniq, counts = np.unique(key, return_counts=True)
        a, b = uniq // state.n, uniq % state.n
        w = state.effective_weight(ti.decay_rate)[a, b] + counts * ti.reinforcement
        state.weight[a, b] = state.weight[b, a] = w
        state.last_contact[a, b] = state.last_contact[b, a] = t
        state.weights_changed()
    for a, b in zip(lo[~tied].tolist(), hi[~tied].tolist()):
        if state.adjacency[a, b]:
            state.weight[a, b] += ti.reinforcement
            state.weight[b, a] = state.weight[a, b]
            continue
        deg = state.adjacency[[a, b]].sum(axis=1)
        if deg[0] < state.capacity[a] and deg[1] < state.capacity[b]:
            state.adjacency[a, b] = state.adjacency[b, a] = True
            state.weight[a, b] = state.weight[b, a] = ti.initial_weight + ti.reinforcement
            state.last_contact[a, b] = state.last_contact[b, a] = t
            state.distance_dirty = True
            formed += 1
        state.weights_changed()
    return formed


def _remove(state, a, b):
    state.adjacency[a, b] = state.adjacency[b, a] = False
    state.weight[a, b] = state.weight[b, a] = 0.0
    state.distance_dirty = True
    state.weights_changed()


def drop_refused(state, cfg) -> int:
    """End every tie between two agents who have walked away from each other. Only the
    "experience" refusal ends a relationship; a tie to a low-rated partner under "reputation" is
    left to fade with neglect like any other unused tie."""
    if cfg.encounter.refusal != "experience":
        return 0
    from .encounter import refused
    a, b = np.nonzero(np.triu(state.adjacency & refused(state, cfg), 1))
    if len(a):
        _remove(state, a, b)
    return len(a)


@TIE_RULES.register("none")
class NoSeverance:
    """Ties never break (stages S1 and S2)."""

    def sever(self, state, cfg, rng=None) -> tuple[int, int]:
        """Remove nothing. Returns ``(removed, formed)``, here ``(0, 0)``."""
        return 0, 0


@TIE_RULES.register("absolute")
class AbsoluteSeverance:
    """A tie breaks when its decayed strength falls below ``severance_threshold``, judged on its
    own strength rather than against alternatives."""

    def sever(self, state, cfg, rng=None):
        """Remove every tie below the threshold. Returns ``(removed, formed)``, with ``formed``
        always 0; freed capacity is refilled separately by the rewiring rule."""
        weak = np.triu(state.adjacency & (state.effective_weight(cfg.ties.decay_rate) < cfg.ties.severance_threshold), 1)
        a, b = np.nonzero(weak)
        if len(a):
            _remove(state, a, b)
        return len(a), 0


@TIE_RULES.register("comparative")
class ComparativeSeverance:
    """Each agent swaps its weakest tie for its best-scoring non-neighbour candidate when the
    candidate, scored at ``prospect_weight``, beats it. One swap per agent per round."""

    def sever(self, state, cfg, rng=None):
        from .encounter import KERNELS
        enc, ti = cfg.encounter, cfg.ties
        w = state.effective_weight(ti.decay_rate).copy()
        k = state.kernel(enc)
        status = state.rating[None, :] ** enc.popularity_exponent
        removed = formed = 0
        for i in range(state.n):
            ties = np.flatnonzero(state.adjacency[i])
            cand = np.flatnonzero((k[i] > 0) & ~state.adjacency[i] & (state.distance[i] >= 2))
            if not len(ties) or not len(cand):
                continue
            tie_scores = (w[i, ties] + enc.eps) ** enc.weight_exponent * status[0, ties] * k[i, ties]
            a = ties[np.argmin(tie_scores)]
            free = state.capacity[cand] - state.adjacency[cand].sum(axis=1)
            cand = cand[free > 0]
            if not len(cand):
                continue
            cand_scores = (ti.prospect_weight + enc.eps) ** enc.weight_exponent * status[0, cand] * k[i, cand]
            b = cand[np.argmax(cand_scores)]
            if cand_scores.max() > tie_scores.min():
                _remove(state, i, a)
                state.adjacency[i, b] = state.adjacency[b, i] = True
                state.weight[i, b] = state.weight[b, i] = ti.initial_weight
                state.last_contact[i, b] = state.last_contact[b, i] = state.t
                state.weights_changed()
                removed += 1
                formed += 1
        return removed, formed
