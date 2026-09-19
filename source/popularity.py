"""Popularity ratings: bounded, lagged signals of how much cooperative value an agent returns."""
from __future__ import annotations

import numpy as np

from .registry import Registry

POPULARITY = Registry("popularity rule")


def _strength_share(state, cfg) -> np.ndarray:
    strength = state.strength(cfg.ties.decay_rate)
    top = strength.max()
    return strength / top if top > 0 else np.zeros(state.n)


def _cooperation_share(state, record):
    moves = np.zeros(state.n)
    coop = np.zeros(state.n)
    actors = np.concatenate([record.initiator, record.partner])
    realised = np.concatenate([record.move_i, record.move_j])
    np.add.at(moves, actors, 1)
    np.add.at(coop, actors, realised == 0)
    return moves, coop


@POPULARITY.register("none")
class NoPopularity:
    def initialise(self, state, cfg):
        state.rating[:] = 1.0

    def update(self, state, record, cfg):
        pass


class _Rated:
    def combine(self, kappa, share, cfg):
        raise NotImplementedError

    def initialise(self, state, cfg):
        self._set_rating(state, cfg)

    def _set_rating(self, state, cfg):
        p = cfg.popularity
        state.rating[:] = p.r_min + (p.r_max - p.r_min) * self.combine(state.coop_ema, _strength_share(state, cfg), cfg)

    def signal(self, state, record, cfg):
        moves, coop = _cooperation_share(state, record)
        return moves, np.divide(coop, moves, out=np.zeros(state.n), where=moves > 0)

    def update(self, state, record, cfg):
        moves, x = self.signal(state, record, cfg)
        eta = cfg.popularity.smoothing
        played = moves > 0
        state.coop_ema[played] = (1 - eta) * state.coop_ema[played] + eta * x[played]
        self._set_rating(state, cfg)


@POPULARITY.register("behaviour_x_strength")
class BehaviourTimesStrength(_Rated):
    # A product: a taker's rating collapses however connected it is, while a connected
    # cooperator's rating compounds, which is the concentration mechanism behind H2.
    def combine(self, kappa, share, cfg):
        return kappa * share


@POPULARITY.register("weighted_sum")
class WeightedSum(_Rated):
    def combine(self, kappa, share, cfg):
        a = cfg.popularity.mix
        return a * kappa + (1 - a) * share


@POPULARITY.register("payoff_delivered")
class PayoffDelivered(_Rated):
    def combine(self, kappa, share, cfg):
        return kappa * share

    def signal(self, state, record, cfg):
        moves = np.zeros(state.n)
        delivered = np.zeros(state.n)
        np.add.at(moves, record.initiator, 1)
        np.add.at(moves, record.partner, 1)
        np.add.at(delivered, record.initiator, record.payoff_j)
        np.add.at(delivered, record.partner, record.payoff_i)
        temptation = cfg.game.payoff[0]
        return moves, np.divide(delivered, moves * temptation, out=np.zeros(state.n), where=moves > 0)
