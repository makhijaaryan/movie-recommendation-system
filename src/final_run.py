"""Re-run every FROZEN final configuration for one dataset and one seed (task 6.2).

For each method in experiments/final_config.json this refits the model with the configuration
chosen on seed 42's validation split (no re-tuning), scores the test split and
  * saves per-user results to data/processed/<dataset>[__seed<S>]/per_user/<method>__<role>.parquet
      rating role : u, n, sse, sae          (per-user count / squared error / absolute error)
      ranking role: u, precision@K, recall@K, hit@K, ndcg@K
    (used by src.uncertainty for bootstrap confidence intervals)
  * for seeds other than 42, appends test rows to experiments/results.csv (seed recorded);
    for seed 42 nothing is logged and the recomputed numbers are checked against the logged ones.

Seed comes from the RECSYS_SEED environment variable (see src/config.py); the split for that seed must
exist first:   RECSYS_SEED=43 python -m src.data --dataset ml_25m
Run:           RECSYS_SEED=43 python -m src.final_run --dataset ml_25m
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.baseline import GlobalMean, MeanRatingBaseline, PopularityCount
from src.cf import ItemKNN, UserKNN
from src.config import SEED, processed_dir
from src.eval import RESULTS_CSV, evaluate_topk, log_result
from src.mf import MF
from src.occf import OCCF, get_graphs, positives_csr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
RANK_COLS = [f"{m}@{k}" for k in (5, 10, 20) for m in ("precision", "recall", "hit", "ndcg")]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def job_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def fit_key(spec: dict) -> str:
    """Specs with the same fit key can share one fitted model (CF: only k differs)."""
    m, p = spec["model"], spec.get("params", {})
    if m in ("item_knn", "user_knn"):
        return f"{m}|shrink={p['shrink']}"
    return job_key(spec)


def per_user_rating(model, test: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    pred = np.clip(model.predict(test["u"].to_numpy(), test["i"].to_numpy()), lo, hi)
    err = test["rating"].to_numpy(float) - pred
    g = pd.DataFrame({"u": test["u"].to_numpy(), "se": err ** 2, "ae": np.abs(err)}).groupby("u")
    out = pd.DataFrame({"n": g.size(), "sse": g["se"].sum(), "sae": g["ae"].sum()}).reset_index()
    return out


def fit_model(spec, ctx):
    """Fit the model for `spec` on the train split (shared across roles via ctx cache)."""
    m, p = spec["model"], spec.get("params", {})
    train, val, n_users, n_items, tr = ctx["train"], ctx["val"], ctx["n_users"], ctx["n_items"], ctx["tr"]
    if m == "global_mean":
        return GlobalMean().fit(train, n_users, n_items)
    if m == "mean_rating":
        return MeanRatingBaseline(damping=p["damping"], min_count=50).fit(train, n_users, n_items)
    if m == "popularity_count":
        return PopularityCount().fit(train, n_users, n_items)
    if m == "item_knn":
        cache = str(processed_dir(ctx["dataset"]) / f"itemknn_sim_shrink{int(p['shrink'])}.npz")
        return ItemKNN(k=p["k_rating"], shrink=p["shrink"]).fit(tr, cache=cache)
    if m == "user_knn":
        return UserKNN(k=p["k_rating"], shrink=p["shrink"]).fit(tr)
    if m in ("mf_plain", "mf_biased"):
        return MF(p["k"], p["lr"], p["lam"], p["biased"], p["decay"]).fit(
            train, n_users, n_items, val, fixed_epochs=p["epochs"], verbose=False)
    if m in ("occf_wrmf", "occf_dual"):
        lu, li = p.get("lam_u", 0.0), p.get("lam_i", 0.0)
        pos = positives_csr(train, n_users, n_items)
        G = get_graphs(ctx["dataset"], pos, p.get("top_m", 20), lu > 0, li > 0)
        return OCCF(p["k"], p["alpha"], p["lam"], lu, li, p["iters"]).fit(
            pos, G.get("user"), G.get("item"), verbose=False)
    raise ValueError(m)


def set_k(model, spec, role):
    """CF only: choose the neighbourhood size for the role (rating vs ranking)."""
    if spec["model"] not in ("item_knn", "user_knn"):
        return
    p = spec["params"]
    kk = p["k_rating"] if role == "rating" else p["k_rank"]
    if spec["model"] == "item_knn":
        model.set_k(kk)
    elif spec["model"] == "user_knn":
        model.k = kk
        model.sk_cache = None


def logged_params(spec: dict) -> dict:
    p = dict(spec.get("params", {}))
    if spec["model"] in ("item_knn", "user_knn"):
        p["eval_users"] = "all"
    if spec["model"] == "mean_rating":
        p["min_count"] = 50
    if spec["model"].startswith("occf"):
        p.setdefault("top_m", 20)
        p.setdefault("lam_u", 0.0)
        p.setdefault("lam_i", 0.0)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these method labels")
    ap.add_argument("--results", default=str(RESULTS_CSV), help="results CSV to log to / compare with")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    methods = cfg["datasets"][a.dataset]
    lo, hi = (0.5, 5.0) if a.dataset.startswith("ml") else (1.0, 5.0)

    train, val, test = (data.load_split(a.dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(a.dataset)
    n_users, n_items = tr.shape
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    ctx = dict(dataset=a.dataset, train=train, val=val, n_users=n_users, n_items=n_items, tr=tr)
    out_dir = processed_dir(a.dataset) / "per_user"
    out_dir.mkdir(parents=True, exist_ok=True)

    # one job per distinct spec; remember which (label, role) pairs it serves
    jobs: dict[str, dict] = {}
    for label, spec in methods.items():
        if a.only and label not in a.only:
            continue
        for role in ("rating", "ranking"):
            if role in spec:
                j = jobs.setdefault(job_key(spec[role]), dict(spec=spec[role], uses=[], roles=set()))
                j["uses"].append((label, role))
                j["roles"].add(role)
    order = sorted(jobs.values(), key=lambda j: fit_key(j["spec"]))
    print(f"seed {SEED}, {a.dataset}: {len(order)} jobs", flush=True)

    model, model_key, logged = None, None, []
    for j in order:
        spec, t0 = j["spec"], time.time()
        if fit_key(spec) != model_key:
            model = None                                   # free memory before the next fit
            model = fit_model(spec, ctx)
            model_key = fit_key(spec)
            print(f"  fitted {model_key[:70]} in {time.time()-t0:.0f}s", flush=True)
        metrics: dict = {}
        if "rating" in j["roles"]:
            set_k(model, spec, "rating")
            pu = per_user_rating(model, test, lo, hi)
            metrics.update(rmse=float(np.sqrt(pu.sse.sum() / pu.n.sum())), mae=float(pu.sae.sum() / pu.n.sum()))
            for label, role in j["uses"]:
                if role == "rating":
                    pu.to_parquet(out_dir / f"{slug(label)}__rating.parquet", index=False)
        if "ranking" in j["roles"]:
            set_k(model, spec, "ranking")
            res, pk = evaluate_topk(model.score_fn, test, trval, n_items, return_per_user=True)
            metrics.update(res)
            for label, role in j["uses"]:
                if role == "ranking":
                    pk[["u"] + RANK_COLS].to_parquet(out_dir / f"{slug(label)}__ranking.parquet", index=False)
        logged.append((spec, metrics))
        print(f"  {spec['model']:<16} {json.dumps(spec.get('params', {}))[:60]:<60} "
              + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k in ("rmse", "ndcg@10", "hit@10"))
              + f"  [{time.time()-t0:.0f}s]", flush=True)

    # seed 42: compare with the rows already logged; other seeds: log new rows
    res_path = Path(a.results)
    if SEED == 42:
        prev = pd.read_csv(res_path)
        prev = prev[(prev.split == "test") & (prev.dataset == a.dataset) & (prev.seed == 42)]
        worst = 0.0
        for spec, metrics in logged:
            c = prev[prev.model == spec["model"]]
            c = c[[all(k in json.loads(r) and np.isclose(json.loads(r)[k], v) for k, v in spec.get("params", {}).items())
                   for r in c.params]]
            if len(c) != 1:
                print(f"  (no unique logged row to compare for {spec})")
                continue
            for k, v in metrics.items():
                if k in c.columns and k != "n_users" and pd.notna(c.iloc[0][k]):
                    worst = max(worst, abs(float(c.iloc[0][k]) - v))
        print(f"seed 42 reproducibility: max |recomputed - logged| over all metrics = {worst:.2e}")
    else:
        for spec, metrics in logged:
            m = {k: v for k, v in metrics.items()}
            log_result(a.dataset, spec["model"], "test", logged_params(spec), m, seed=SEED,
                       notes="final multi-seed (no re-tuning)", path=res_path)
        print(f"logged {len(logged)} test rows for seed {SEED}")


if __name__ == "__main__":
    main()
