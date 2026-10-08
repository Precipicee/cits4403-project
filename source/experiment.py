"""Experiments: cells of configurations, repeated seeds, parallel runs and aggregation.

An experiment is declared where it is used, so its settings are visible next to its results::

    exp = Experiment(name="cities", base="M4", root_seed=20260917, n_seeds=25,
                     axes={"population.giver_fraction": [0.0, 0.5, 1.0]})
    tables = run(exp)
"""
from __future__ import annotations

import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import io
from .instruments import CORE_METRICS, FLOW_METRICS, STRUCTURE_METRICS

TAIL = 100          # rounds averaged for flow metrics and steady-state checks
THIN = 5            # rounds between points in timeseries_mean.csv


@dataclass
class Experiment:
    """A grid of cities: every combination of the values in ``axes`` (with ``fixed`` overrides on
    top of the ``base`` preset), each run for ``n_seeds`` seeds."""

    name: str
    base: str
    root_seed: int
    n_seeds: int
    axes: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    instruments: list = field(default_factory=lambda: ["core", "structure", "snapshot"])
    write_raw: bool = True

    def cells(self) -> list[dict]:
        """Every combination of axis values, as one override dictionary per cell."""
        keys = list(self.axes)
        return [dict(zip(keys, values)) for values in itertools.product(*self.axes.values())]

    def config(self, cell: dict):
        """Validated configuration for one cell."""
        from .presets import preset
        return preset(self.base, **{**self.fixed, **cell})


def tail_slopes(income_gini: np.ndarray, mean_income: np.ndarray) -> tuple[float, float]:
    """Per-round slopes over the last ``TAIL`` rounds: rolling income Gini, and mean income relative
    to its mean."""
    tail = min(TAIL, len(income_gini))
    x = np.arange(tail)
    g, m = income_gini[-tail:], mean_income[-tail:]
    if np.isnan(g).any() or np.isnan(m).any() or tail < 3:
        return float("nan"), float("nan")
    return float(np.polyfit(x, g, 1)[0]), float(np.polyfit(x, m, 1)[0] / max(abs(m.mean()), 1e-12))


def steady_state(income_gini: np.ndarray, mean_income: np.ndarray) -> bool:
    """Slope of rolling income Gini below 1e-4 per round and slope of mean income below 0.1% of
    mean income per round, over the last ``TAIL`` rounds."""
    slope_g, slope_m = tail_slopes(income_gini, mean_income)
    return bool(abs(slope_g) < 1e-4 and abs(slope_m) < 1e-3)


def summarise(result) -> tuple[dict, dict]:
    """Final values per run, and the core time series thinned to every ``THIN`` rounds."""
    out: dict = {}
    core = result.outputs.get("core")
    if core is not None:
        with warnings.catch_warnings():
            # A class-specific rate is undefined when that class is absent (f = 0 or f = 1).
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in CORE_METRICS:
                v = core[k]
                out[k] = float(np.nanmean(v[-TAIL:])) if k in FLOW_METRICS else float(v[-1])
        out["steady_state"] = steady_state(core["income_gini"], core["mean_income"])
        out["slope_income_gini"], out["slope_mean_income"] = tail_slopes(core["income_gini"], core["mean_income"])
    structure = result.outputs.get("structure")
    if structure is not None:
        for k in STRUCTURE_METRICS:
            out[k] = float(structure[k][-1])
        rounds = structure["round"]
        late = rounds >= rounds.max() - TAIL
        if late.sum() >= 3:
            out["assortativity_slope"] = float(np.polyfit(rounds[late], structure["assortativity_class"][late], 1)[0]) \
                if not np.isnan(structure["assortativity_class"][late]).any() else float("nan")
    thin = {}
    if core is not None:
        idx = np.arange(THIN - 1, len(core["gini"]), THIN)
        thin = {"round": idx + 1, **{k: core[k][idx] for k in CORE_METRICS}}
    return out, thin


def _versions() -> dict:
    import matplotlib
    import networkx
    return {"numpy": np.__version__, "networkx": networkx.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__}


def _run_task(task):
    exp, cell_index, cell, repeat = task
    from .simulation import Simulation
    cfg = exp.config(cell)
    result = Simulation(cfg, exp.root_seed, repeat, tuple(exp.instruments)).run()
    if exp.write_raw:
        io.write_run(io.RAW / exp.name / cfg.config_hash() / f"seed_{repeat}", result, _versions())
    final, thin = summarise(result)
    row = {**cell, "cell": cell_index, "repeat": repeat, "config_hash": cfg.config_hash(), **final}
    return row, thin


def run(exp: Experiment, workers: int | None = None, progress: bool = True) -> dict[str, pd.DataFrame]:
    """Run every cell and seed in parallel, aggregate, and save the tables under
    ``data/results/<name>/``. Returns the tables by name (``runs``, ``cells``, ``timeseries_mean``)."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(exp, c, cell, r) for c, cell in enumerate(exp.cells()) for r in range(exp.n_seeds)]
    workers = workers or os.cpu_count()
    start, rows, thins = time.time(), [], []
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for k, (row, thin) in enumerate(pool.map(_run_task, tasks, chunksize=1), 1):
            rows.append(row)
            thins.append((row["cell"], row["repeat"], thin))
            if progress and (k % max(1, len(tasks) // 10) == 0 or k == len(tasks)):
                print(f"{exp.name}: {k}/{len(tasks)} runs, {time.time() - start:.0f} s", flush=True)
    tables = aggregate(exp, pd.DataFrame(rows), thins)
    io.save_results(exp.name, tables)
    return tables


def aggregate(exp: Experiment, runs: pd.DataFrame, thins=None, n_boot: int = 2000) -> dict[str, pd.DataFrame]:
    """Per-cell summaries of every metric over seeds (mean, median, standard deviation and a
    percentile bootstrap interval), plus mean time series with intervals when ``thins`` is given."""
    from .metrics import bootstrap_ci
    axes = list(exp.axes)
    runs = runs.sort_values(["cell", "repeat"]).reset_index(drop=True)
    metric_cols = [c for c in runs.columns if c not in set(axes) | {"cell", "repeat", "config_hash"}]
    rng = np.random.default_rng(exp.root_seed)
    cell_rows = []
    for cell, group in runs.groupby("cell", sort=True):
        values = group[metric_cols].astype(float).to_numpy()
        lo, hi = bootstrap_ci(values, n_boot=n_boot, rng=rng)
        for m_i, metric in enumerate(metric_cols):
            col = values[:, m_i]
            cell_rows.append({**{a: group[a].iloc[0] for a in axes}, "cell": cell, "metric": metric,
                              "mean": np.nanmean(col), "median": np.nanmedian(col), "sd": np.nanstd(col, ddof=1),
                              "ci_low": lo[m_i], "ci_high": hi[m_i], "n": int(np.isfinite(col).sum())})
    tables = {"runs": runs, "cells": pd.DataFrame(cell_rows)}
    if thins:
        ts_rows = []
        by_cell: dict = {}
        for cell, repeat, thin in thins:
            by_cell.setdefault(cell, []).append(thin)
        cells = exp.cells()
        for cell, items in sorted(by_cell.items()):
            rounds = items[0]["round"]
            stack = np.stack([np.column_stack([t[k] for k in CORE_METRICS]) for t in items])
            mean = np.nanmean(stack, axis=0)
            lo, hi = bootstrap_ci(stack, n_boot=500, rng=rng)
            for r_i, rnd in enumerate(rounds):
                row = {**cells[cell], "cell": cell, "round": int(rnd)}
                for m_i, metric in enumerate(CORE_METRICS):
                    row[metric] = mean[r_i, m_i]
                    row[f"{metric}_lo"], row[f"{metric}_hi"] = lo[r_i, m_i], hi[r_i, m_i]
                ts_rows.append(row)
        tables["timeseries_mean"] = pd.DataFrame(ts_rows)
    return tables


def results(exp: "Experiment", rerun: bool = False, workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Aggregated tables for ``exp``; runs it when ``rerun`` is set or when no results are saved."""
    if rerun or not (io.RESULTS / exp.name / "cells.csv").exists():
        return run(exp, workers=workers)
    return io.results(exp.name)
