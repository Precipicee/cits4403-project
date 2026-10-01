"""Model configuration: frozen dataclasses, validation, dotted-key overrides and hashing."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

GIVER_STRATEGIES = ("allc", "tft", "gtft", "tf2t", "grudger", "pavlov")
TAKER_STRATEGIES = ("alld", "random", "joss")
# Joss is available to every city but not drawn by default: the default taker class is the two
# unconditional defectors, and Joss is mixed in only where a scenario asks for it.
DEFAULT_TAKERS = ("alld", "random")


class ConfigError(ValueError):
    """Raised when a configuration is invalid; lists every violation."""


@dataclass(frozen=True)
class PopulationConfig:
    n_agents: int = 200
    giver_fraction: float = 0.5
    giver_composition: dict = field(default_factory=lambda: {k: 1.0 for k in GIVER_STRATEGIES})
    taker_composition: dict = field(default_factory=lambda: {k: 1.0 for k in DEFAULT_TAKERS})
    placement: str = "random"
    initial_wealth: str = "zero"
    endowment_mean: float = 10.0
    endowment_gini: float = 0.2
    style_mix: float | None = None
    k_broad: int = 10
    k_narrow: int = 3


@dataclass(frozen=True)
class TopologyConfig:
    name: str = "erdos_renyi"
    mean_degree: int = 6
    ws_rewire_prob: float = 0.1


@dataclass(frozen=True)
class GameConfig:
    # Axelrod's tournament payoffs (T, R, P, S): 2R > T + S makes mutual cooperation the
    # wealth-maximising pairing, so city wealth is a readout of how agents treat each other.
    payoff: tuple = (5.0, 3.0, 1.0, 0.0)
    rho: float = 1.0
    subsistence_cost: float = 0.0
    noise: float = 0.0


@dataclass(frozen=True)
class StrategyParams:
    gtft_forgiveness: float = 0.1
    random_cooperation: float = 0.5
    # Axelrod's Joss: Tit-for-Tat that slips in an unprovoked defection one game in ten, so it
    # cooperates ~90% of the time and holds a rating close to a giver's while still exploiting.
    joss_defection: float = 0.1
    classes: dict = field(default_factory=lambda: {**{k: "giver" for k in GIVER_STRATEGIES},
                                                   **{k: "taker" for k in TAKER_STRATEGIES}})


@dataclass(frozen=True)
class EncounterConfig:
    rule: str = "sampled"
    initiated: int = 1
    max_received: int | None = 4
    on_saturated: str = "redraw"
    redraw_attempts: int = 3
    d_max: int = 3
    kernel: str = "power"
    # Link lengths in geo-located social ties follow roughly P(d) ~ 1/d, hence lambda = 1.
    distance_exponent: float = 1.0
    weight_exponent: float = 1.0
    popularity_exponent: float = 1.0
    eps: float = 0.1
    # Who an agent will not play. "experience": a pair stops meeting once either side has
    # defected on the other at their last game (walking away instead of retaliating).
    # "reputation": nobody plays a partner rated below refusal_threshold. "none": every
    # meeting is accepted.
    refusal: str = "none"
    refusal_memory: int | None = None
    refusal_threshold: float = 0.25


@dataclass(frozen=True)
class TieConfig:
    initial_weight: float = 1.0
    reinforcement: float = 1.0
    # Neglect, not punishment: an untouched tie of weight 1 falls below theta = 0.1
    # after ln(10) / 0.05 ~ 46 weeks.
    decay_rate: float = 0.05
    severance_threshold: float = 0.1
    severance_rule: str = "absolute"
    prospect_weight: float = 1.0
    capacity: str = "initial_degree"
    capacity_k: int = 6


@dataclass(frozen=True)
class RewiringConfig:
    rule: str = "local_refill"
    fallback: str = "global"
    every: int = 1


@dataclass(frozen=True)
class PopularityConfig:
    rule: str = "behaviour_x_strength"
    r_min: float = 0.1
    r_max: float = 1.0
    smoothing: float = 0.1
    initial_cooperation: float = 0.5
    mix: float = 0.5


@dataclass(frozen=True)
class UpdateConfig:
    order: str = "synchronous"


@dataclass(frozen=True)
class MetricsConfig:
    income_window: int = 25
    structure_every: int = 10
    community_method: str = "louvain"
    assortativity_attribute: str = "class"
    inequality: str = "gini"


@dataclass(frozen=True)
class RunConfig:
    n_rounds: int = 500


@dataclass(frozen=True)
class ModelConfig:
    population: PopulationConfig = field(default_factory=PopulationConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    game: GameConfig = field(default_factory=GameConfig)
    strategies: StrategyParams = field(default_factory=StrategyParams)
    encounter: EncounterConfig = field(default_factory=EncounterConfig)
    ties: TieConfig = field(default_factory=TieConfig)
    rewiring: RewiringConfig = field(default_factory=RewiringConfig)
    popularity: PopularityConfig = field(default_factory=PopularityConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ---- serialisation -------------------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        parts = {}
        for f in dataclasses.fields(cls):
            sub_cls = f.default_factory().__class__
            values = dict(d.get(f.name, {}))
            if "payoff" in values:
                values["payoff"] = tuple(values["payoff"])
            parts[f.name] = sub_cls(**values)
        return cls(**parts).validate()

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        return cls.from_dict(json.loads(text))

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    # ---- overrides -----------------------------------------------------------------
    def override(self, changes: dict[str, Any]) -> "ModelConfig":
        """Return a validated copy with dotted keys replaced, e.g. {"encounter.d_max": 2}."""
        grouped: dict[str, dict] = {}
        for key, value in changes.items():
            section, _, name = key.partition(".")
            if not hasattr(self, section) or not name or not hasattr(getattr(self, section), name):
                raise ConfigError(f"unknown configuration key: {key}")
            if name == "payoff":
                value = tuple(value)
            grouped.setdefault(section, {})[name] = value
        replaced = {s: dataclasses.replace(getattr(self, s), **v) for s, v in grouped.items()}
        return dataclasses.replace(self, **replaced).validate()

    def get(self, key: str) -> Any:
        section, _, name = key.partition(".")
        return getattr(getattr(self, section), name)

    # ---- validation ----------------------------------------------------------------
    def validate(self) -> "ModelConfig":
        e = []
        p, t, g, s = self.population, self.topology, self.game, self.strategies
        enc, ti, rw, po = self.encounter, self.ties, self.rewiring, self.popularity
        T, R, P, S = g.payoff
        if not (T > R > P > S):
            e.append("payoff must satisfy T > R > P > S")
        if not (2 * R > T + S):
            e.append("payoff must satisfy 2R > T + S")
        if p.n_agents < 4:
            e.append("population.n_agents must be at least 4")
        if not 0.0 <= p.giver_fraction <= 1.0:
            e.append("population.giver_fraction must lie in [0, 1]")
        for name, comp, allowed in (("giver", p.giver_composition, GIVER_STRATEGIES),
                                    ("taker", p.taker_composition, TAKER_STRATEGIES)):
            if not comp or any(k not in allowed or w < 0 for k, w in comp.items()) or sum(comp.values()) <= 0:
                e.append(f"population.{name}_composition must weight {allowed} with non-negative weights")
        if p.placement not in ("random", "clustered"):
            e.append("population.placement must be random or clustered")
        if p.initial_wealth not in ("zero", "unequal"):
            e.append("population.initial_wealth must be zero or unequal")
        if not 0.0 <= p.endowment_gini < 0.5:
            e.append("population.endowment_gini must lie in [0, 0.5)")
        if t.name not in ("complete", "erdos_renyi", "watts_strogatz", "barabasi_albert"):
            e.append(f"unknown topology.name: {t.name}")
        if t.name in ("watts_strogatz", "barabasi_albert") and t.mean_degree % 2:
            e.append("topology.mean_degree must be even for watts_strogatz and barabasi_albert")
        if not 0.0 <= s.joss_defection <= 1.0:
            e.append("strategies.joss_defection must lie in [0, 1]")
        if g.rho <= 0 or not 0.0 <= g.noise <= 1.0 or g.subsistence_cost < 0:
            e.append("game.rho > 0, game.noise in [0, 1], game.subsistence_cost >= 0 required")
        if enc.rule not in ("sampled", "all_edges"):
            e.append("encounter.rule must be sampled or all_edges")
        if enc.initiated < 1 or enc.d_max < 1:
            e.append("encounter.initiated and encounter.d_max must be at least 1")
        if enc.max_received is not None and enc.max_received < 1:
            e.append("encounter.max_received must be None or at least 1")
        if enc.on_saturated not in ("redraw", "drop"):
            e.append("encounter.on_saturated must be redraw or drop")
        if enc.kernel not in ("power", "rank"):
            e.append("encounter.kernel must be power or rank")
        if enc.refusal not in ("none", "experience", "reputation"):
            e.append("encounter.refusal must be none, experience or reputation")
        if enc.refusal_memory is not None and enc.refusal_memory < 1:
            e.append("encounter.refusal_memory must be None or at least 1")
        if enc.eps <= 0:
            e.append("encounter.eps must be positive")
        if ti.decay_rate < 0 or ti.severance_threshold < 0 or ti.initial_weight <= 0 or ti.reinforcement <= 0:
            e.append("ties: decay_rate >= 0, severance_threshold >= 0, weights > 0 required")
        if ti.severance_rule not in ("absolute", "comparative", "none"):
            e.append("ties.severance_rule must be absolute, comparative or none")
        if ti.capacity not in ("initial_degree", "uniform", "unbounded"):
            e.append("ties.capacity must be initial_degree, uniform or unbounded")
        if rw.rule not in ("none", "local_refill") or rw.fallback not in ("global", "none") or rw.every < 1:
            e.append("rewiring: rule none|local_refill, fallback global|none, every >= 1")
        if po.rule not in ("none", "behaviour_x_strength", "weighted_sum", "payoff_delivered"):
            e.append(f"unknown popularity.rule: {po.rule}")
        if not 0 < po.r_min <= po.r_max or not 0 < po.smoothing <= 1 or not 0 <= po.mix <= 1:
            e.append("popularity: 0 < r_min <= r_max, 0 < smoothing <= 1, 0 <= mix <= 1 required")
        if self.update.order not in ("synchronous", "asynchronous"):
            e.append("update.order must be synchronous or asynchronous")
        if self.run.n_rounds < 1 or self.metrics.income_window < 1 or self.metrics.structure_every < 1:
            e.append("run.n_rounds, metrics.income_window and metrics.structure_every must be at least 1")
        if e:
            raise ConfigError("; ".join(e))
        return self
