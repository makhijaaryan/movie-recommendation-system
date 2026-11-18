"""Learning curves of the frozen final models (task 6.7), saved to reports/tables/learning_curves.csv.

* MF (biased, the RMSE configuration): train RMSE (first 2M training ratings) and validation RMSE after each epoch.
* One-class dual-regularized (ranking configuration): validation NDCG@10 (10,000 sampled users) after each iteration.
Both are refit on the train split exactly as in the final runs (seed from RECSYS_SEED).

Run:  python -m src.curves --dataset netflix   (and ml_25m)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.eval import evaluate_topk
from src.mf import MF
from src.occf import OCCF, get_graphs, positives_csr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"


def run(dataset: str) -> pd.DataFrame:
    methods = json.loads(CONFIG.read_text())["datasets"][dataset]
    train, val, _ = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    nu, ni = tr.shape
    rows = []

    p = methods["MF (biased)"]["rating"]["params"]
    mf = MF(p["k"], p["lr"], p["lam"], p["biased"], p["decay"]).fit(train, nu, ni, val, fixed_epochs=p["epochs"],
                                                                    verbose=False)
    for h in mf.history:
        rows.append(dict(dataset=dataset, model="MF (biased)", step=h["epoch"], train_rmse=h["train_rmse"],
                         val_rmse=h["val_rmse"]))
    print(f"{dataset}: MF curve with {len(mf.history)} epochs", flush=True)

    p = methods["One-class dual-regularized"]["ranking"]["params"]
    pos = positives_csr(train, nu, ni)
    G = get_graphs(dataset, pos, 20, True, True)
    cb = lambda m, it: evaluate_topk(m.score_fn, val, tr, ni, max_users=10000)["ndcg@10"]
    oc = OCCF(p["k"], p["alpha"], p["lam"], p["lam_u"], p["lam_i"], p["iters"]).fit(
        pos, G["user"], G["item"], callback=cb, verbose=False)
    for h in oc.history:
        rows.append(dict(dataset=dataset, model="One-class dual-regularized", step=h["iter"], val_ndcg10=h["score"]))
    print(f"{dataset}: one-class curve with {len(oc.history)} iterations", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"learning_curves_{dataset}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    run(ap.parse_args().dataset)
