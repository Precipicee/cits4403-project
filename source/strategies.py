"""The eight Iterated Prisoner's Dilemma strategies, vectorised over interaction sides.

Moves are integers: 0 = cooperate (C), 1 = defect (D). For an actor i facing partner j,
the partner's previous move is ``last_move[j, i]`` and the one before it ``prev_move[j, i]``.
"""
from __future__ import annotations

import numpy as np

from .registry import Registry

STRATEGIES = Registry("strategy")
C, D, NONE = 0, 1, -1


class Strategy:
    key: str = ""
    stochastic: bool = False

    def moves(self, state, actors, partners, params, rng) -> np.ndarray:
        raise NotImplementedError


@STRATEGIES.register("allc")
class AlwaysCooperate(Strategy):
    def moves(self, state, actors, partners, params, rng):
        return np.zeros(len(actors), dtype=np.int8)


@STRATEGIES.register("alld")
class AlwaysDefect(Strategy):
    def moves(self, state, actors, partners, params, rng):
        return np.ones(len(actors), dtype=np.int8)


@STRATEGIES.register("tft")
class TitForTat(Strategy):
    def moves(self, state, actors, partners, params, rng):
        last = state.last_move[partners, actors]
        return np.where(last == NONE, C, last).astype(np.int8)


@STRATEGIES.register("gtft")
class GenerousTitForTat(Strategy):
    stochastic = True

    def moves(self, state, actors, partners, params, rng):
        m = TitForTat.moves(self, state, actors, partners, params, rng)
        defected = np.flatnonzero(m == D)
        if len(defected):
            forgive = rng.random(len(defected)) < params.gtft_forgiveness
            m[defected[forgive]] = C
        return m


@STRATEGIES.register("tf2t")
class TitForTwoTats(Strategy):
    def moves(self, state, actors, partners, params, rng):
        two = (state.last_move[partners, actors] == D) & (state.prev_move[partners, actors] == D)
        return two.astype(np.int8)


@STRATEGIES.register("grudger")
class Grudger(Strategy):
    def moves(self, state, actors, partners, params, rng):
        return state.grudge[actors, partners].astype(np.int8)


@STRATEGIES.register("pavlov")
class Pavlov(Strategy):
    # Win-stay, lose-shift: a C from the partner (payoff R or T) is a win.
    def moves(self, state, actors, partners, params, rng):
        own = state.last_move[actors, partners]
        opp = state.last_move[partners, actors]
        stay = np.where(opp == C, own, 1 - own)
        return np.where(own == NONE, C, stay).astype(np.int8)


@STRATEGIES.register("random")
class RandomStrategy(Strategy):
    stochastic = True

    def moves(self, state, actors, partners, params, rng):
        return (rng.random(len(actors)) >= params.random_cooperation).astype(np.int8)


STRATEGY_KEYS = tuple(STRATEGIES.names())
STRATEGY_ID = {k: i for i, k in enumerate(STRATEGY_KEYS)}
STRATEGY_OBJECTS = tuple(STRATEGIES.create(k) for k in STRATEGY_KEYS)
