"""The simulation loop: one city observed over T rounds."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encounter import ENCOUNTERS, distance_matrix
from .game import InteractionRecord, empty_record, play, update_memory
from .popularity import POPULARITY
from .rewiring import REWIRING
from .state import build_city, make_streams
from .ties import TIE_RULES, reinforce


@dataclass
class RoundInfo:
    record: InteractionRecord
    formed: int = 0
    removed: int = 0
    refilled: int = 0


@dataclass
class RunResult:
    config: object
    root_seed: int
    repeat: int
    outputs: dict = field(default_factory=dict)
    state: object = None
    rng_state: dict | None = None


def _concat(records: list[InteractionRecord]) -> InteractionRecord:
    # Counted over every record: an agent that found no partner contributes idle but no game.
    idle, dropped = sum(r.idle for r in records), sum(r.dropped for r in records)
    played = [r for r in records if r.n_games] or [empty_record()]
    cat = lambda name: np.concatenate([getattr(r, name) for r in played])  # noqa: E731
    return InteractionRecord(cat("initiator"), cat("partner"), cat("move_i"), cat("move_j"),
                             cat("payoff_i"), cat("payoff_j"), cat("had_history"), cat("had_tie"),
                             idle, dropped)


class Simulation:
    def __init__(self, cfg, root_seed: int = 0, repeat: int = 0, instruments=("core",)):
        from .instruments import make_instruments
        self.cfg = cfg
        self.root_seed, self.repeat = root_seed, repeat
        self.streams = make_streams(root_seed, repeat)
        self.state = build_city(cfg, self.streams)
        self.encounter = ENCOUNTERS.create(cfg.encounter.rule)
        self.tie_rule = TIE_RULES.create(cfg.ties.severance_rule)
        self.rewiring = REWIRING.create(cfg.rewiring.rule)
        self.popularity = POPULARITY.create(cfg.popularity.rule)
        self.instruments = make_instruments(instruments, cfg)
        for inst in self.instruments:
            inst.start(self.state, cfg, self.streams.metrics)

    # ---- one game step (payoffs, memory, ties) ----------------------------------------
    def _settle(self, record: InteractionRecord) -> int:
        st, cfg = self.state, self.cfg
        gain = np.zeros(st.n)
        if record.n_games:
            np.add.at(gain, record.initiator, cfg.game.rho * record.payoff_i)
            np.add.at(gain, record.partner, cfg.game.rho * record.payoff_j)
            np.add.at(st.games_played, record.initiator, 1)
            np.add.at(st.games_played, record.partner, 1)
            np.add.at(st.coop_moves, record.initiator, record.move_i == 0)
            np.add.at(st.coop_moves, record.partner, record.move_j == 0)
        st.wealth += gain
        st.payoff_total += gain
        st.income += gain
        update_memory(st, record)
        return reinforce(st, record, cfg)

    def run_round(self) -> RoundInfo:
        st, cfg, rng = self.state, self.cfg, self.streams.dynamics
        st.t += 1
        st.received[:] = 0
        st.income[:] = 0
        if st.distance_dirty:
            st.distance = distance_matrix(st.adjacency, cfg.encounter.d_max)
            st.distance_dirty = False
        if cfg.update.order == "synchronous":
            pairs, idle, dropped = self.encounter.pairs(st, cfg, rng)
            record = play(st, pairs, cfg, rng, idle, dropped)
            formed = self._settle(record)
        else:
            parts, formed = [], 0
            for i in rng.permutation(st.n):
                pairs, idle, dropped = self.encounter.pairs(st, cfg, rng, agents=[i])
                part = play(st, pairs, cfg, rng, idle, dropped)
                formed += self._settle(part)
                parts.append(part)
            record = _concat(parts)
        st.wealth -= cfg.game.subsistence_cost
        st.income -= cfg.game.subsistence_cost
        st.income_buffer[(st.t - 1) % len(st.income_buffer)] = st.income
        removed = refilled = 0
        if st.t % cfg.rewiring.every == 0:
            removed, swapped = self.tie_rule.sever(st, cfg, rng)
            formed += swapped
            refilled = self.rewiring.refill(st, cfg, rng)
        self.popularity.update(st, record, cfg)
        info = RoundInfo(record, formed, removed, refilled)
        for inst in self.instruments:
            inst.update(st, info, self.streams.metrics)
        return info

    def run(self, keep_state: bool = False) -> RunResult:
        for _ in range(self.cfg.run.n_rounds):
            self.run_round()
        outputs = {inst.name: inst.result() for inst in self.instruments}
        return RunResult(self.cfg, self.root_seed, self.repeat, outputs,
                         state=self.state if keep_state else None,
                         rng_state=self.streams.dynamics.bit_generator.state)
