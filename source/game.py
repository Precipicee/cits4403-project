"""Playing one round of Prisoner's Dilemma games for a set of pairs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strategies import STRATEGY_OBJECTS


@dataclass(frozen=True)
class InteractionRecord:
    initiator: np.ndarray
    partner: np.ndarray
    move_i: np.ndarray        # realised moves after noise
    move_j: np.ndarray
    payoff_i: np.ndarray      # before rho scaling
    payoff_j: np.ndarray
    had_history: np.ndarray
    had_tie: np.ndarray
    idle: int = 0
    dropped: int = 0

    @property
    def n_games(self) -> int:
        return len(self.initiator)


def payoff_table(payoff) -> np.ndarray:
    """``table[own, other]`` is the payoff to a player moving ``own`` against ``other``."""
    T, R, P, S = payoff
    return np.array([[R, S], [T, P]], dtype=float)


def empty_record(idle: int = 0, dropped: int = 0) -> InteractionRecord:
    """A round in which nothing was played. The counters still travel: an agent that found no
    partner, or whose requests were all dropped, produces no game but is not nothing."""
    e = np.zeros(0, dtype=np.int64)
    f = np.zeros(0)
    b = np.zeros(0, dtype=bool)
    return InteractionRecord(e, e, e.astype(np.int8), e.astype(np.int8), f, f, b, b, idle, dropped)


def play(state, pairs: np.ndarray, cfg, rng, idle: int = 0, dropped: int = 0) -> InteractionRecord:
    """Play one game per row of ``pairs`` using the memory in ``state`` (not updated here)."""
    if len(pairs) == 0:
        return empty_record(idle, dropped)
    i, j = pairs[:, 0], pairs[:, 1]
    actors = np.concatenate([i, j])
    partners = np.concatenate([j, i])
    moves = np.empty(len(actors), dtype=np.int8)
    sid = state.strategy[actors]
    for s_id, strategy in enumerate(STRATEGY_OBJECTS):
        mask = np.flatnonzero(sid == s_id)
        if len(mask):
            moves[mask] = strategy.moves(state, actors[mask], partners[mask], cfg.strategies, rng)
    if cfg.game.noise > 0:
        flip = rng.random(len(moves)) < cfg.game.noise
        moves[flip] = 1 - moves[flip]
    n = len(i)
    mi, mj = moves[:n], moves[n:]
    table = payoff_table(cfg.game.payoff)
    return InteractionRecord(
        initiator=i, partner=j, move_i=mi, move_j=mj,
        payoff_i=table[mi, mj], payoff_j=table[mj, mi],
        had_history=state.last_move[i, j] != -1, had_tie=state.adjacency[i, j].copy(),
        idle=idle, dropped=dropped,
    )


def update_memory(state, record: InteractionRecord) -> None:
    """Write realised moves into memory. When a directed pair played twice, the later game's
    move becomes ``last_move`` and the earlier one ``prev_move``."""
    if record.n_games == 0:
        return
    actors = np.concatenate([record.initiator, record.partner])
    partners = np.concatenate([record.partner, record.initiator])
    moves = np.concatenate([record.move_i, record.move_j])
    order = np.concatenate([np.arange(record.n_games), np.arange(record.n_games)])
    order = np.argsort(order, kind="stable")
    actors, partners, moves = actors[order], partners[order], moves[order]
    key = actors.astype(np.int64) * state.n + partners
    _, first_idx, counts = np.unique(key, return_index=True, return_counts=True)
    single = first_idx[counts == 1]
    a, p = actors[single], partners[single]
    state.prev_move[a, p] = state.last_move[a, p]
    state.last_move[a, p] = moves[single]
    for idx in first_idx[counts > 1]:
        k = key[idx]
        for m in moves[key == k]:
            state.prev_move[actors[idx], partners[idx]] = state.last_move[actors[idx], partners[idx]]
            state.last_move[actors[idx], partners[idx]] = m
    # Grudger remembers any defection it has suffered.
    defected = moves == 1
    state.grudge[partners[defected], actors[defected]] = True
