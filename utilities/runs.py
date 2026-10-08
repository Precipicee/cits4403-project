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
from typing import NamedTuple

import pandas as pd

import numpy as np

from source.experiment import summarise
from source.presets import stage
from source.simulation import Simulation
from source.strategies import STRATEGY_KEYS

DEFAULT_SEED = 20260918


def limit_threads() -> None:
    """One BLAS thread per process. Only takes effect if called before numpy is imported."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


class SweepResult(NamedTuple):
    """A sweep kept with its histories: one row per run, plus one row per recorded round."""
    runs: pd.DataFrame
    core: pd.DataFrame
    structure: pd.DataFrame


def run_one(scenario: str, base: str, overrides: dict, root_seed: int, repeat: int,
            instruments=("core", "structure"), timeseries: bool = False):
    """One city: the scenario label, its axis values and the final metrics. With ``timeseries``
    the round-by-round histories come back too, rather than being summarised away."""
    cfg = stage(base, **overrides)
    result = Simulation(cfg, root_seed=root_seed, repeat=repeat, instruments=instruments).run()
    metrics, thin = summarise(result)
    label = {"scenario": scenario, "repeat": repeat,
             "giver_fraction": cfg.population.giver_fraction, "topology": cfg.topology.name}
    row = {**label, **metrics}
    if result.outputs.get("snapshot") is not None:
        row.update(by_strategy(result.outputs["snapshot"]))
    if not timeseries:
        return row
    history = {"core": {**label, **thin}}
    if result.outputs.get("structure") is not None:
        history["structure"] = {**label, **result.outputs["structure"]}
    return row, history


def by_strategy(snap: dict) -> dict:
    """Final wealth, games played, rating and the share left with no ties, per strategy present,
    as columns such as ``wealth_alld``."""
    out = {}
    isolated = snap["adjacency"].sum(axis=1) == 0
    for s_id, key in enumerate(STRATEGY_KEYS):
        mask = snap["strategy"] == s_id
        if mask.any():
            out[f"wealth_{key}"] = float(snap["wealth"][mask].mean())
            out[f"games_{key}"] = float(snap["games_played"][mask].mean())
            out[f"rating_{key}"] = float(snap["rating"][mask].mean())
            out[f"isolated_{key}"] = float(isolated[mask].mean())
    return out


def _task(args):
    return run_one(*args)


def _frame(history: dict) -> pd.DataFrame:
    """One history dict to a tidy frame: array fields become columns, labels are broadcast."""
    arrays = {k: v for k, v in history.items() if hasattr(v, "__len__") and not isinstance(v, str)}
    labels = {k: v for k, v in history.items() if k not in arrays}
    return pd.DataFrame(arrays).assign(**labels)


def sweep(scenarios: dict[str, tuple[str, dict]], fractions, n_seeds: int = 5,
          root_seed: int = DEFAULT_SEED, instruments=("core", "structure"),
          workers: int | None = None, timeseries: bool = False, **shared):
    """Every scenario x giver fraction x seed, run in parallel.

    ``scenarios`` maps a label to ``(stage name, overrides)``; ``shared`` holds overrides applied
    to every scenario, such as ``{"population.n_agents": 100}``. Seeds are the same across
    scenarios, so differences come from the rules rather than from the draw.

    Returns one row per run, or a :class:`SweepResult` of (runs, core, structure) when
    ``timeseries`` is set, where the two histories carry one row per recorded round.
    """
    tasks = [(label, base, {**shared, **overrides, "population.giver_fraction": f}, root_seed,
              repeat, instruments, timeseries)
             for label, (base, overrides) in scenarios.items()
             for f in fractions for repeat in range(n_seeds)]
    out = _map(tasks, workers)
    if not timeseries:
        return pd.DataFrame(out)
    runs = pd.DataFrame([row for row, _ in out])
    frames = lambda key: pd.concat([_frame(h[key]) for _, h in out if key in h], ignore_index=True)  # noqa: E731
    return SweepResult(runs, frames("core"), frames("structure"))


def _map(tasks, workers: int | None) -> list[dict]:
    return parallel_map(_task, tasks, workers)


def parallel_map(func, tasks, workers: int | None = None) -> list:
    """``[func(t) for t in tasks]``, in parallel where the platform allows it.

    ``fork`` keeps this usable from a notebook, where a spawned worker cannot re-import the
    interactive session; a function defined in a notebook cell can be passed directly.
    """
    if "fork" not in mp.get_all_start_methods():
        return [func(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count(),
                             mp_context=mp.get_context("fork")) as pool:
        return list(pool.map(func, tasks, chunksize=1))


def cell_means(runs: pd.DataFrame, metrics, by=("scenario", "giver_fraction")) -> pd.DataFrame:
    """Mean over seeds for each metric, with the standard error of that mean."""
    grouped = runs.groupby(list(by))[list(metrics)]
    mean, sem = grouped.mean(), grouped.sem()
    return mean.join(sem, rsuffix="_se").reset_index()


def stage_deltas(runs: pd.DataFrame, metric: str, order, by=("giver_fraction", "repeat")) -> pd.DataFrame:
    """Paired change in ``metric`` from each scenario in ``order`` to the next.

    Runs are paired on the seed and the giver fraction, and seeds are shared across scenarios,
    so each difference isolates the one mechanism the later stage adds rather than mixing in
    the luck of the draw. Returns the mean change per step with its standard error.
    """
    wide = runs.pivot_table(index=list(by), columns="scenario", values=metric)
    rows = []
    for a, b in zip(list(order)[:-1], list(order)[1:]):
        d = (wide[b] - wide[a]).dropna()
        rows.append({"step": f"{a.split()[0]} \u2192 {b.split()[0]}", "from": a, "to": b,
                     "mean": d.mean(), "se": d.sem(), "n": int(len(d))})
    return pd.DataFrame(rows)


def series_means(history: pd.DataFrame, metrics, by=("scenario",)) -> pd.DataFrame:
    """Mean over seeds at each recorded round, with the standard error of that mean.

    The same summary as :func:`cell_means`, grouped by round as well, which is what the
    over-time figures plot as a line and a band.
    """
    return cell_means(history, metrics, by=list(by) + ["round"])
