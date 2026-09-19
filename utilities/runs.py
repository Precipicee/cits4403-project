"""Small parallel sweeps held in memory: one row per run, nothing written to disk.

The full experiment pipeline in ``source.experiment`` reads a configuration file and saves
results; this is the lightweight path used inside the notebook, where every run is cheap
enough to repeat on the spot.

Call :func:`limit_threads` before numpy is imported. The workers run one city each, so a
multi-threaded BLAS underneath them costs several times the runtime in contention.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from source.experiment import summarise
from source.presets import stage
from source.simulation import Simulation

DEFAULT_SEED = 20260918


def limit_threads() -> None:
    """One BLAS thread per process. Only takes effect if called before numpy is imported."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_one(scenario: str, base: str, overrides: dict, root_seed: int, repeat: int,
            instruments=("core", "structure")) -> dict:
    """One city: returns the scenario label, its axis values and the final metrics."""
    cfg = stage(base, **overrides)
    result = Simulation(cfg, root_seed=root_seed, repeat=repeat, instruments=instruments).run()
    metrics, _ = summarise(result)
    return {"scenario": scenario, "repeat": repeat,
            "giver_fraction": cfg.population.giver_fraction, "topology": cfg.topology.name,
            **metrics}


def _task(args):
    return run_one(*args)


def sweep(scenarios: dict[str, tuple[str, dict]], fractions, n_seeds: int = 5,
          root_seed: int = DEFAULT_SEED, instruments=("core", "structure"),
          workers: int | None = None, **shared) -> pd.DataFrame:
    """Every scenario x giver fraction x seed, run in parallel.

    ``scenarios`` maps a label to ``(stage name, overrides)``; ``shared`` holds overrides applied
    to every scenario, such as ``{"population.n_agents": 100}``. Seeds are the same across
    scenarios, so differences come from the rules rather than from the draw.
    """
    tasks = [(label, base, {**shared, **overrides, "population.giver_fraction": f}, root_seed, repeat, instruments)
             for label, (base, overrides) in scenarios.items()
             for f in fractions for repeat in range(n_seeds)]
    return pd.DataFrame(_map(tasks, workers))


def _map(tasks, workers: int | None) -> list[dict]:
    """Run the tasks in parallel where the platform allows it, otherwise one at a time.

    ``fork`` keeps this usable from a notebook, where a spawned worker cannot re-import the
    interactive session.
    """
    if "fork" not in mp.get_all_start_methods():
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count(),
                             mp_context=mp.get_context("fork")) as pool:
        return list(pool.map(_task, tasks, chunksize=1))


def cell_means(runs: pd.DataFrame, metrics, by=("scenario", "giver_fraction")) -> pd.DataFrame:
    """Mean over seeds for each metric, with the standard error of that mean."""
    grouped = runs.groupby(list(by))[list(metrics)]
    mean, sem = grouped.mean(), grouped.sem()
    return mean.join(sem, rsuffix="_se").reset_index()
