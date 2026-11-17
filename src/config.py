"""Run-wide settings shared by all modules.

RECSYS_SEED (environment variable, default 42) sets the random seed used for the data split, the
Netflix user sample and model initialisation. Seed 42 uses data/processed/<dataset>/ as before; any
other seed uses data/processed/<dataset>__seed<seed>/, so multi-seed runs never overwrite the main data.

    RECSYS_SEED=43 python -m src.data --dataset ml_25m
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = int(os.environ.get("RECSYS_SEED", "42"))


def processed_dir(name: str) -> Path:
    return PROC / name if SEED == 42 else PROC / f"{name}__seed{SEED}"
