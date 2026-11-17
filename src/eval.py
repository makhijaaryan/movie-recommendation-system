"""Shared evaluation for every model (proposal §5).

Rating prediction: RMSE, MAE.
Top-K ranking:     Precision@K, Recall@K, Hit Rate@K, NDCG@K (binary relevance).

Protocol (identical for all models and both datasets):
  * relevant item for user u = a held-out rating >= RELEVANT_THRESHOLD in the evaluated split
  * candidates = all items except those the user already interacted with in
    the excluded matrix (train for validation; train + val for test)
  * only users with >= 1 relevant item are scored
  * models supply `score_fn(user_indices) -> (n_users, n_items) score array`
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "experiments" / "results.csv"

RELEVANT_THRESHOLD = 4.0
KS = (5, 10, 20)
# User-history buckets by number of TRAIN ratings (used for the low-history analysis).
BUCKET_EDGES = (0, 30, 60, 120, 300, np.inf)
BUCKET_LABELS = ("<30", "30-59", "60-119", "120-299", "300+")


# ------------------------------------------------------------------ rating metrics
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating(predict_fn, df: pd.DataFrame, rating_range=(0.5, 5.0)) -> dict:
    """predict_fn(u_idx, i_idx) -> predicted ratings. Predictions are clipped to the valid range."""
    pred = np.clip(predict_fn(df["u"].to_numpy(), df["i"].to_numpy()), *rating_range)
    return {"rmse": rmse(df["rating"], pred), "mae": mae(df["rating"], pred), "n": len(df)}


# --------------------------------------------------------------------- top-K metrics
def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (ties broken by lower index)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    idx = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, idx, axis=1)


def topk_metrics_from_lists(topk: np.ndarray, rel: np.ndarray, ks=KS) -> dict[str, np.ndarray]:
    """Per-user metrics given ranked item lists and a dense boolean relevance matrix.

    topk: (b, Kmax) ranked item indices; rel: (b, n_items) bool. Users must have >=1 relevant.
    """
    n_rel = rel.sum(axis=1).astype(float)
    hits = np.take_along_axis(rel, topk, axis=1).astype(float)          # (b, Kmax)
    discounts = 1.0 / np.log2(np.arange(2, topk.shape[1] + 2))
    out: dict[str, np.ndarray] = {}
    for k in ks:
        h = hits[:, :k]
        nh = h.sum(axis=1)
        dcg = (h * discounts[:k]).sum(axis=1)
        ideal_n = np.minimum(n_rel, k).astype(int)
        idcg = np.array([discounts[:n].sum() for n in ideal_n])
        out[f"precision@{k}"] = nh / k
        out[f"recall@{k}"] = nh / n_rel
        out[f"hit@{k}"] = (nh > 0).astype(float)
        out[f"ndcg@{k}"] = dcg / idcg
    return out


def evaluate_topk(score_fn, eval_df: pd.DataFrame, exclude: sp.csr_matrix, n_items: int,
                  ks=KS, threshold: float = RELEVANT_THRESHOLD, batch_size: int = 2000,
                  max_users: int | None = None, seed: int = 42,
                  return_per_user: bool = False):
    """Top-K evaluation.

    eval_df:  rows (u, i, rating) of the split being evaluated (val or test)
    exclude:  CSR matrix of interactions to hide from the candidates (train, or train+val)
    max_users: optional random subsample of eligible users (seeded) to save time
    Returns dict of mean metrics (+ 'n_users'); with return_per_user also a per-user DataFrame.
    """
    rel_df = eval_df[eval_df["rating"] >= threshold]
    users = np.sort(rel_df["u"].unique())
    if max_users is not None and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    rel_csr = sp.csr_matrix(
        (np.ones(len(rel_df), dtype=bool), (rel_df["u"].to_numpy(), rel_df["i"].to_numpy())),
        shape=(exclude.shape[0], n_items))
    kmax = max(ks)
    parts: dict[str, list[np.ndarray]] = {}
    for s in range(0, len(users), batch_size):
        b = users[s:s + batch_size]
        scores = np.asarray(score_fn(b), dtype=np.float32).copy()
        seen = exclude[b].tocoo()
        scores[seen.row, seen.col] = -np.inf
        rel = rel_csr[b].toarray()
        rel &= np.isfinite(scores)        # relevant items already seen can't be recommended
        keep = rel.any(axis=1)
        b, scores, rel = b[keep], scores[keep], rel[keep]   # drop users with nothing recommendable
        if len(b) == 0:
            continue
        m = topk_metrics_from_lists(_topk(scores, kmax), rel, ks)
        m["u"] = b
        for k_, v in m.items():
            parts.setdefault(k_, []).append(v)
    per_user = pd.DataFrame({k_: np.concatenate(v) for k_, v in parts.items()})
    result = {c: float(per_user[c].mean()) for c in per_user.columns if c != "u"}
    result["n_users"] = int(len(per_user))
    return (result, per_user) if return_per_user else result


def add_history_bucket(per_user: pd.DataFrame, train_csr: sp.csr_matrix) -> pd.DataFrame:
    """Attach each user's train-rating count and history bucket to a per-user metrics frame."""
    counts = np.diff(train_csr.indptr)[per_user["u"].to_numpy()]
    out = per_user.copy()
    out["n_train"] = counts
    out["bucket"] = pd.cut(counts, bins=list(BUCKET_EDGES), labels=list(BUCKET_LABELS), right=False)
    return out


# -------------------------------------------------------------------------- logging
def log_result(dataset: str, model: str, split: str, params: dict, metrics: dict,
               seed: int = 42, notes: str = "", path: Path = RESULTS_CSV) -> None:
    """Append one experiment row to experiments/results.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": dataset, "model": model,
           "split": split, "seed": seed, "params": json.dumps(params, sort_keys=True),
           "notes": notes, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}}
    new = not path.exists()
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        missing = [k for k in row if k not in header]
        if missing:  # rewrite with the union of columns
            df = pd.read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(path, index=False)
            return
        row = {k: row.get(k, "") for k in header}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
