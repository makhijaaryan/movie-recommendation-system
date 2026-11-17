"""Non-personalized baselines (proposal §4.2).

GlobalMean        : predicts the training mean for every rating (RMSE reference).
MeanRatingBaseline: ranks items by average rating, requiring a minimum number of
                    ratings (proposal wording) and/or damping toward the global
                    mean:  score_i = (sum_i + m * mu) / (n_i + m).
PopularityCount   : ranks items by number of training ratings (extra reference;
                    logged as DIFFERENCE.md D14).
Rating prediction for the mean baseline is the item's damped mean.

Run from the project root:  python -m src.baseline --dataset ml_small
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.eval import KS, evaluate_rating, evaluate_topk, log_result

M_GRID = (0, 5, 10, 25, 50, 100, 250)       # damping strengths tried on the validation set
MIN_COUNT = 50                               # proposal: "requiring a minimum number of ratings"


class GlobalMean:
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        self.mu = float(train["rating"].mean())
        return self

    def predict(self, u, i):
        return np.full(len(u), self.mu)


class MeanRatingBaseline:
    def __init__(self, damping: float = 0.0, min_count: int = 0):
        self.damping, self.min_count = damping, min_count

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        self.mu = float(train["rating"].mean())
        n = np.bincount(train["i"], minlength=n_items).astype(float)
        s = np.bincount(train["i"], weights=train["rating"], minlength=n_items)
        self.n = n
        self.item_score = (s + self.damping * self.mu) / (n + self.damping + 1e-12)
        self.item_score[n == 0] = self.mu
        self.rank_score = np.where(n >= self.min_count, self.item_score, -np.inf)
        # items below the min count still need an order below all qualifying ones
        below = n < self.min_count
        self.rank_score[below] = self.item_score[below] - 1e3
        return self

    def predict(self, u, i):
        return self.item_score[i]

    def score_fn(self, users):
        return np.broadcast_to(self.rank_score, (len(users), len(self.rank_score)))


class PopularityCount:
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        self.rank_score = np.bincount(train["i"], minlength=n_items).astype(float)
        return self

    def score_fn(self, users):
        return np.broadcast_to(self.rank_score, (len(users), len(self.rank_score)))


def run(dataset: str, max_users: int | None = None) -> pd.DataFrame:
    rng = (0.5, 5.0) if dataset.startswith("ml") else (1.0, 5.0)
    train, val, test = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr_csr = data.load_train_csr(dataset)
    n_users, n_items = tr_csr.shape
    trval_csr = tr_csr + sp.csr_matrix(
        (np.ones(len(val)), (val["u"], val["i"])), shape=tr_csr.shape)   # hide val items on test
    rows = []

    def record(model, split, params, metrics, notes=""):
        log_result(dataset, model, split, params, metrics, notes=notes)
        rows.append({"model": model, "split": split, **params, **metrics})

    # Rating prediction + ranking on validation to choose the damping m
    gm = GlobalMean().fit(train, n_users, n_items)
    for name, df in (("val", val), ("test", test)):
        record("global_mean", name, {}, evaluate_rating(gm.predict, df, rng))

    best = None
    for m in M_GRID:
        b = MeanRatingBaseline(damping=m, min_count=MIN_COUNT).fit(train, n_users, n_items)
        r = evaluate_rating(b.predict, val, rng)
        t = evaluate_topk(b.score_fn, val, tr_csr, n_items, max_users=max_users)
        record("mean_rating", "val", {"damping": m, "min_count": MIN_COUNT}, {**r, **t})
        if best is None or r["rmse"] < best[0]:
            best = (r["rmse"], m)
    m_star = best[1]                                   # chosen by validation RMSE
    b = MeanRatingBaseline(damping=m_star, min_count=MIN_COUNT).fit(train, n_users, n_items)
    r = evaluate_rating(b.predict, test, rng)
    t = evaluate_topk(b.score_fn, test, trval_csr, n_items, max_users=max_users)
    record("mean_rating", "test", {"damping": m_star, "min_count": MIN_COUNT}, {**r, **t},
           notes="damping chosen by val RMSE")

    pc = PopularityCount().fit(train, n_users, n_items)
    record("popularity_count", "val", {}, evaluate_topk(pc.score_fn, val, tr_csr, n_items, max_users=max_users))
    record("popularity_count", "test", {}, evaluate_topk(pc.score_fn, test, trval_csr, n_items, max_users=max_users))
    return pd.DataFrame(rows)


def run_rank_selected(dataset: str) -> dict:
    """Test-evaluate the mean-rating baseline at the damping with the best VALIDATION NDCG@10
    (the RMSE-selected damping, used by `run`, gives the weakest ranking). Logged for the final table."""
    import json
    from src.eval import RESULTS_CSV
    df = pd.read_csv(RESULTS_CSV)
    v = df[(df.dataset == dataset) & (df.model == "mean_rating") & (df.split == "val")].copy()
    v["damping"] = v.params.apply(lambda s: json.loads(s)["damping"])
    m_star = float(v.loc[v["ndcg@10"].idxmax(), "damping"])
    train, val, test = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr_csr = data.load_train_csr(dataset)
    n_users, n_items = tr_csr.shape
    trval = tr_csr + sp.csr_matrix((np.ones(len(val)), (val["u"], val["i"])), shape=tr_csr.shape)
    b = MeanRatingBaseline(damping=m_star, min_count=MIN_COUNT).fit(train, n_users, n_items)
    t = evaluate_topk(b.score_fn, test, trval, n_items)
    log_result(dataset, "mean_rating", "test", {"damping": m_star, "min_count": MIN_COUNT}, t,
               notes="final-rank damping chosen by val NDCG@10")
    return {"damping": m_star, **t}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(data.CONFIG), required=True)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--rank-final", action="store_true", help="test-evaluate mean-rating at the NDCG-selected damping")
    a = ap.parse_args()
    if a.rank_final:
        print(run_rank_selected(a.dataset))
        raise SystemExit
    pd.set_option("display.width", 250, "display.max_columns", 30)
    out = run(a.dataset, a.max_users)
    cols = ["model", "split", "damping", "rmse", "mae", "precision@10", "recall@10", "hit@10", "ndcg@10", "n_users"]
    print(out[[c for c in cols if c in out]].round(4).to_string(index=False))
