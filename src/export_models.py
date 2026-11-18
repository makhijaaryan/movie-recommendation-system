"""Export the frozen MovieLens-25M models that the web demo serves (ranking configurations, train split only).

Writes data/processed/ml_25m/serving/{mf.npz, occf.npz}. Item CF, user CF and popularity are rebuilt from the
training matrix when the API starts (item similarities come from the existing cache).

Run:  python -m src.export_models
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import data
from src.config import processed_dir
from src.final_run import fit_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
DATASET = "ml_25m"


def main() -> None:
    methods = json.loads(CONFIG.read_text())["datasets"][DATASET]
    train, val = data.load_split(DATASET, "train"), data.load_split(DATASET, "val")
    tr = data.load_train_csr(DATASET)
    nu, ni = tr.shape
    ctx = dict(dataset=DATASET, train=train, val=val, n_users=nu, n_items=ni, tr=tr)
    out = processed_dir(DATASET) / "serving"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    spec = methods["MF (biased)"]["ranking"]
    m = fit_model(spec, ctx)
    np.savez(out / "mf.npz", Q=m.Q, bi=m.bi, mu=float(m.mu), lam=float(m.lam), biased=bool(m.biased))
    print(f"MF exported ({time.time() - t0:.0f}s)", flush=True)

    spec = methods["One-class dual-regularized"]["ranking"]
    m = fit_model(spec, ctx)
    np.savez(out / "occf.npz", Q=m.Q, alpha=float(m.alpha), lam=float(m.lam))
    print(f"One-class exported ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
