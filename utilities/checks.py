"""Assertions used in notebook check cells."""
from __future__ import annotations

import numpy as np


def check(condition: bool, message: str) -> None:
    """Raise with ``message`` if ``condition`` is false; otherwise print a one-line pass."""
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def trace(sim) -> dict:
    """Run a simulation round by round, keeping what exact comparisons need."""
    wealth = []
    for _ in range(sim.cfg.run.n_rounds):
        sim.run_round()
        wealth.append(sim.state.wealth.copy())
    s = sim.state
    return {"wealth": np.array(wealth), "adjacency": s.adjacency.copy(), "weight": s.weight.copy(),
            "last_move": s.last_move.copy(), "rng": sim.streams.dynamics.bit_generator.state}


def same_run(a: dict, b: dict) -> bool:
    """True when two traced runs match exactly: wealth every round, final ties, tie weights and
    move memory, and the state of the random generator."""
    return (np.array_equal(a["wealth"], b["wealth"]) and np.array_equal(a["adjacency"], b["adjacency"])
            and np.array_equal(a["weight"], b["weight"]) and np.array_equal(a["last_move"], b["last_move"])
            and a["rng"] == b["rng"])
