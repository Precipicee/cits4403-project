"""Instruments record metrics as a simulation runs."""
from __future__ import annotations

import numpy as np

from . import metrics as M
from .encounter import refused
from .registry import Registry
from .strategies import STRATEGY_KEYS

INSTRUMENTS = Registry("instrument")

CORE_METRICS = (
    "mean_wealth", "median_wealth", "total_wealth", "mean_income", "gini", "top10_share",
    "bottom50_share", "income_gini", "theil", "theil_within", "theil_between",
    "coop_rate", "coop_rate_giver", "coop_rate_taker", "cc_share", "cd_share", "dd_share",
    "payoff_gap", "repeat_share", "stranger_share", "n_ties", "mean_weight", "turnover",
    "largest_component", "isolated", "idle", "dropped", "max_received", "mean_rating",
    "games_giver", "games_taker", "wealth_giver", "wealth_taker", "gini_games", "gini_surplus", "closed_pairs",
)
STRUCTURE_METRICS = (
    "assortativity_class", "assortativity_strategy", "modularity", "purity", "n_communities",
    "degree_giver", "degree_taker", "weight_giver", "weight_taker",
    "realised_degree_giver", "realised_degree_taker", "isolated_giver", "isolated_taker",
    "clustering", "clustering_giver", "clustering_taker", "transitivity",
    "mean_path_length", "n_components",
)
# Rates are summarised by their mean over the last 100 rounds; stocks by their final value.
FLOW_METRICS = {"mean_income", "income_gini", "coop_rate", "coop_rate_giver", "coop_rate_taker",
                "cc_share", "cd_share", "dd_share", "repeat_share", "stranger_share", "turnover",
                "idle", "dropped", "max_received"}


def _class_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else float("nan")


@INSTRUMENTS.register("core")
class Core:
    """Every round: prosperity, inequality, cooperation, tie and component measures, listed in
    ``CORE_METRICS``. Cheap enough to run on every city."""

    name = "core"

    def start(self, state, cfg, rng):
        self.cfg = cfg
        self.data = {k: np.full(cfg.run.n_rounds, np.nan) for k in CORE_METRICS}

    def update(self, st, info, rng):
        d, r, t, cfg = self.data, info.record, st.t - 1, self.cfg
        w = st.wealth
        d["mean_wealth"][t], d["median_wealth"][t], d["total_wealth"][t] = w.mean(), np.median(w), w.sum()
        d["mean_income"][t] = st.income.mean()
        d["gini"][t], d["top10_share"][t], d["bottom50_share"][t] = M.gini(w), M.top_share(w), M.bottom_share(w)
        d["income_gini"][t] = M.gini(st.income_buffer[: min(st.t, len(st.income_buffer))].sum(axis=0))
        d["theil"][t], d["theil_within"][t], d["theil_between"][t] = M.theil_decomposition(w, st.giver)
        n_games = r.n_games
        if n_games:
            moves_giver = np.concatenate([st.giver[r.initiator], st.giver[r.partner]])
            moves = np.concatenate([r.move_i, r.move_j]) == 0
            d["coop_rate"][t] = moves.mean()
            d["coop_rate_giver"][t] = moves[moves_giver].mean() if moves_giver.any() else np.nan
            d["coop_rate_taker"][t] = moves[~moves_giver].mean() if (~moves_giver).any() else np.nan
            both = (r.move_i == 0) & (r.move_j == 0)
            neither = (r.move_i == 1) & (r.move_j == 1)
            d["cc_share"][t], d["dd_share"][t] = both.mean(), neither.mean()
            d["cd_share"][t] = 1 - both.mean() - neither.mean()
            d["repeat_share"][t] = r.had_history.mean()
            d["stranger_share"][t] = 1 - r.had_tie.mean()
        per_game = np.divide(st.payoff_total, st.games_played, out=np.full(st.n, np.nan), where=st.games_played > 0)
        d["payoff_gap"][t] = _class_mean(per_game, ~st.giver & (st.games_played > 0)) - _class_mean(per_game, st.giver & (st.games_played > 0))
        n_ties = st.adjacency.sum() / 2
        d["n_ties"][t] = n_ties
        d["mean_weight"][t] = st.effective_weight(cfg.ties.decay_rate)[st.adjacency].mean() if n_ties else np.nan
        d["turnover"][t] = (info.removed + info.formed + info.refilled) / n_ties if n_ties else np.nan
        labels = M.components(st.adjacency)
        d["largest_component"][t] = np.bincount(labels).max() / st.n
        d["isolated"][t] = (st.adjacency.sum(axis=1) == 0).sum()
        d["idle"][t], d["dropped"][t] = r.idle, r.dropped
        d["max_received"][t] = st.received.max()
        d["mean_rating"][t] = st.rating.mean()
        # Games played is attention received and given; with every payoff positive, wealth is
        # close to a multiple of it, so its Gini is the positional part of wealth inequality.
        d["games_giver"][t] = _class_mean(st.games_played, st.giver)
        d["games_taker"][t] = _class_mean(st.games_played, ~st.giver)
        d["wealth_giver"][t] = _class_mean(w, st.giver)
        d["wealth_taker"][t] = _class_mean(w, ~st.giver)
        d["gini_games"][t] = M.gini(st.games_played)
        # Surplus over mutual defection: each game's payoff less P, so the payoffs become
        # (T-P, R-P, 0, S-P) and being exploited costs wealth. Nothing in the model reads wealth,
        # so this re-measures the same runs rather than changing them.
        d["gini_surplus"][t] = M.gini(w - cfg.game.rho * cfg.game.payoff[2] * st.games_played)
        closed = refused(st, cfg)
        d["closed_pairs"][t] = 0 if closed is None else np.triu(closed, 1).sum()

    def result(self):
        return self.data


@INSTRUMENTS.register("structure")
class Structure:
    """At round 0, every ``structure_every`` rounds and at the end: assortativity, communities, degree and tie
    strength by class, clustering and path length, listed in ``STRUCTURE_METRICS``. Slower than
    ``Core``, so it is recorded less often."""

    name = "structure"

    def start(self, state, cfg, rng):
        self.cfg = cfg
        self.rounds, self.rows = [], []
        self._record(state, rng)

    def _record(self, st, rng):
        cfg = self.cfg
        g = st.to_networkx(cfg.ties.decay_rate)
        seed = int(rng.integers(2**32))
        q, purity, n_comm = M.modularity_and_purity(g, st.giver, cfg.metrics.community_method, seed)
        deg = st.adjacency.sum(axis=1)
        w = st.effective_weight(cfg.ties.decay_rate)
        mean_w = np.divide((w * st.adjacency).sum(axis=1), deg, out=np.full(st.n, np.nan), where=deg > 0)
        realised = (st.last_game >= st.t - cfg.metrics.income_window + 1).sum(axis=1)
        iso = deg == 0
        local_c, transitivity = M.clustering(st.adjacency)
        self.rounds.append(st.t)
        self.rows.append({
            "assortativity_class": M.assortativity(g, "giver"),
            "assortativity_strategy": M.assortativity(g, "strategy"),
            "modularity": q, "purity": purity, "n_communities": n_comm,
            "degree_giver": _class_mean(deg, st.giver), "degree_taker": _class_mean(deg, ~st.giver),
            "weight_giver": float(np.nanmean(mean_w[st.giver])) if (st.giver & ~iso).any() else np.nan,
            "weight_taker": float(np.nanmean(mean_w[~st.giver])) if (~st.giver & ~iso).any() else np.nan,
            "realised_degree_giver": _class_mean(realised, st.giver),
            "realised_degree_taker": _class_mean(realised, ~st.giver),
            "isolated_giver": int((iso & st.giver).sum()), "isolated_taker": int((iso & ~st.giver).sum()),
            "clustering": float(local_c.mean()),
            "clustering_giver": _class_mean(local_c, st.giver),
            "clustering_taker": _class_mean(local_c, ~st.giver),
            "transitivity": transitivity,
            "mean_path_length": M.mean_path_length(st.adjacency),
            "n_components": M.n_components(st.adjacency),
        })

    def update(self, st, info, rng):
        if st.t % self.cfg.metrics.structure_every == 0 or st.t == self.cfg.run.n_rounds:
            self._record(st, rng)

    def result(self):
        out = {"round": np.array(self.rounds)}
        for k in STRUCTURE_METRICS:
            out[k] = np.array([row[k] for row in self.rows], dtype=float)
        return out


def _snapshot(st, cfg):
    p, l = M.lorenz(st.wealth)
    return {"round": st.t, "strategy": st.strategy.copy(), "giver": st.giver.copy(),
            "wealth": st.wealth.copy(), "adjacency": st.adjacency.copy(),
            "weight": st.effective_weight(cfg.ties.decay_rate).copy(), "rating": st.rating.copy(),
            "coop_ema": st.coop_ema.copy(), "games_played": st.games_played.copy(),
            "coop_moves": st.coop_moves.copy(), "lorenz_p": p, "lorenz_l": l}


@INSTRUMENTS.register("snapshot")
class Snapshot:
    """The whole city at the final round, for network drawings and per-strategy summaries."""

    name = "snapshot"

    def start(self, state, cfg, rng):
        self.cfg, self.data = cfg, None

    def update(self, st, info, rng):
        if st.t == self.cfg.run.n_rounds:
            self.data = _snapshot(st, self.cfg)

    def result(self):
        return self.data


@INSTRUMENTS.register("snapshots")
class Snapshots:
    """The whole city at round 0, every ``snapshot_every`` rounds and at the end, for drawings
    and animations over time."""

    name = "snapshots"

    def start(self, state, cfg, rng):
        self.cfg, self.data = cfg, [_snapshot(state, cfg)]

    def update(self, st, info, rng):
        if st.t % self.cfg.metrics.snapshot_every == 0 or st.t == self.cfg.run.n_rounds:
            self.data.append(_snapshot(st, self.cfg))

    def result(self):
        return self.data


def make_instruments(names, cfg):
    """A fresh instrument for each registered name, e.g. ``("core", "structure")``."""
    return [INSTRUMENTS.create(n) for n in names]


__all__ = ["CORE_METRICS", "STRUCTURE_METRICS", "FLOW_METRICS", "STRATEGY_KEYS", "make_instruments"]
