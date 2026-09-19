"""Readers and writers for run outputs and aggregated result tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS, RAW = DATA / "results", DATA / "raw"


def write_run(path: Path, result, versions: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = {"config": result.config.to_dict(), "root_seed": result.root_seed,
            "repeat": result.repeat, "versions": versions}
    (path / "config.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    if "core" in result.outputs:
        df = pd.DataFrame(result.outputs["core"])
        df.insert(0, "round", np.arange(1, len(df) + 1))
        df.to_csv(path / "timeseries.csv.gz", index=False, float_format="%.6g")
    if "structure" in result.outputs:
        pd.DataFrame(result.outputs["structure"]).to_csv(path / "structure.csv.gz", index=False, float_format="%.6g")
    if result.outputs.get("snapshot") is not None:
        # Stored unpacked, so it matches the shape the plural "snapshots" instrument writes.
        np.savez_compressed(path / "snapshot.npz", **result.outputs["snapshot"])
    if result.outputs.get("snapshots"):
        frames = result.outputs["snapshots"]
        stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
        np.savez_compressed(path / "snapshots.npz", **stacked)


def read_snapshots(path: Path) -> list[dict]:
    with np.load(path) as z:
        stacked = {k: z[k] for k in z.files}
    return [{k: v[i] for k, v in stacked.items()} for i in range(len(stacked["round"]))]


def results(name: str) -> dict[str, pd.DataFrame]:
    folder = RESULTS / name
    return {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}


def save_results(name: str, tables: dict[str, pd.DataFrame]) -> Path:
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    for stem, df in tables.items():
        df.to_csv(folder / f"{stem}.csv", index=False, float_format="%.6g")
    return folder
