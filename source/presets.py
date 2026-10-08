"""Stage presets M0 to M5. Each stage adds one mechanism to the one before it."""
from __future__ import annotations

from .config import ModelConfig

# Chosen by the calibration sweep: the settings closest to the starting values that keep a 50/50
# city between collapse and saturation, with ties turning over and the network in one piece.
CALIBRATED = {"encounter.eps": 0.1, "ties.decay_rate": 0.05,
              "ties.severance_threshold": 0.1, "encounter.max_received": 4}

_M0 = {
    "topology.name": "complete",
    "encounter.d_max": 1, "encounter.distance_exponent": 0.0, "encounter.weight_exponent": 0.0,
    "encounter.popularity_exponent": 0.0, "encounter.max_received": None,
    "ties.decay_rate": 0.0, "ties.severance_rule": "none",
    "rewiring.rule": "none", "popularity.rule": "none",
}
_M1 = {**_M0, "topology.name": "erdos_renyi"}
_M2 = {**_M1, "ties.decay_rate": ModelConfig().ties.decay_rate, "ties.severance_rule": "absolute"}
_M3 = {**_M2, "encounter.d_max": 3, "encounter.distance_exponent": 1.0, "encounter.weight_exponent": 1.0,
       "encounter.popularity_exponent": 1.0, "encounter.max_received": 4,
       "popularity.rule": "behaviour_x_strength"}
_M4 = {**_M3, "rewiring.rule": "local_refill"}
_M5 = {**_M4, "encounter.refusal": "experience"}

STAGES = {"M0": _M0, "M1": _M1, "M1-edges": {**_M1, "encounter.rule": "all_edges"},
          "M2": _M2, "M3": _M3, "M4": _M4, "M5": _M5}


def stage(name: str, **overrides) -> ModelConfig:
    """Stage preset with optional dotted-key overrides, e.g. ``stage("M2", **{"ties.decay_rate": 0})``."""
    return ModelConfig().override({**STAGES[name], **overrides})


def calibrated(**overrides) -> ModelConfig:
    return stage("M4", **{**CALIBRATED, **overrides})


def preset(name: str, **overrides) -> ModelConfig:
    return calibrated(**overrides) if name == "calibrated" else stage(name, **overrides)
