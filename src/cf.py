"""Neighborhood collaborative filtering (proposal §4.3): user-based and item-based.

Similarity: cosine on mean-centred ratings (users centred by their own mean, i.e.
adjusted cosine for items / Pearson-like for users), multiplied by a significance
weight  n_co / (n_co + shrink)  where n_co is the number of co-rating users/items.

Two outputs from the same neighbourhoods:
  * rating prediction:  r_hat(u,i) = mu_u + sum_j s * (r_j - mu_j) / sum_j |s|
  * ranking score:      sum_j s * (r_j - mu_j)   (unnormalised: rewards evidence;
                        the normalised version ranks items backed by one neighbour first)
where j runs over the k nearest neighbours (items j rated by u, or users j who rated i).
Only positive similarities are kept.

Run from the project root, e.g.
    python -m src.cf --dataset ml_small --model item --mode tune
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.config import SEED
from src.eval import evaluate_rating, evaluate_topk, log_result

MAX_NEIGHBORS = 300          # neighbours stored per item (item-based)
SHRINK = 25.0                # significance-weighting constant (DIFFERENCE.md D18)


def center(train: sp.csr_matrix):
    """Return (user_means, centred CSR, binary CSR)."""
    train = train.tocsr().astype(np.float32)
    counts = np.diff(train.indptr)
    sums = np.asarray(train.sum(axis=1)).ravel()
    mu = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).astype(np.float32)
    rc = train.copy()
    rc.data = rc.data - np.repeat(mu, counts)
    binary = train.copy()
    binary.data = np.ones_like(binary.data)
    return mu, rc, binary


def _topk_rows(dense: np.ndarray, k: int):
    """Top-k column indices/values per row, descending."""
    k = min(k, dense.shape[1])
    part = np.argpartition(-dense, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(dense, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _predict_pairs(model, u: np.ndarray, i: np.ndarray, batch: int) -> np.ndarray:
    """Vectorised rating prediction for arbitrary (u, i) pairs, processed per user batch."""
    order = np.argsort(u, kind="stable")
    us, is_ = u[order], i[order]
    out = np.empty(len(u), dtype=np.float32)
    users = np.unique(us)
    for s in range(0, len(users), batch):
        bu = users[s:s + batch]
        lo, hi = np.searchsorted(us, bu[0], "left"), np.searchsorted(us, bu[-1], "right")
        num, den = model._num_den(bu)
        rows = np.searchsorted(bu, us[lo:hi])
        n, d = num[rows, is_[lo:hi]], den[rows, is_[lo:hi]]
        pred = model.mu[us[lo:hi]] + np.where(d > 1e-9, n / np.maximum(d, 1e-9), 0.0)
        out[lo:hi] = pred
    res = np.empty_like(out)
    res[order] = out
    return res


# ------------------------------------------------------------------------------ item-based
class ItemKNN:
    def __init__(self, k: int = 50, shrink: float = SHRINK):
        self.k, self.shrink = k, shrink

    def fit(self, train: sp.csr_matrix, cache: str | None = None, threads: int = 4):
        self.mu, self.rc, self.bin = center(train)
        self.n_users, self.n_items = train.shape
        if cache and os.path.exists(cache):
            z = np.load(cache)
            self.nb_idx, self.nb_val = z["idx"], z["val"]
        else:
            self.nb_idx, self.nb_val = self._build_similarity(threads)
            if cache:
                np.savez(cache, idx=self.nb_idx, val=self.nb_val)
        self.set_k(self.k)
        return self

    def _build_similarity(self, threads: int, block: int = 256):
        rc_t = self.rc.T.tocsr()                       # items x users (centred)
        b_t = self.bin.T.tocsr()                        # items x users (binary)
        norm = np.sqrt(np.asarray(rc_t.multiply(rc_t).sum(axis=1)).ravel()).astype(np.float32)
        norm[norm == 0] = 1.0
        n_items = self.n_items
        width = min(MAX_NEIGHBORS, n_items - 1)
        idx = np.zeros((n_items, width), dtype=np.int32)
        val = np.zeros((n_items, width), dtype=np.float32)

        def work(start: int):
            stop = min(start + block, n_items)
            d = np.ascontiguousarray(rc_t[start:stop].toarray().T)        # users x block
            g = rc_t @ d                                                  # items x block
            # co-rating indicator must come from the binary matrix: a centred rating can be exactly 0
            db = np.ascontiguousarray(b_t[start:stop].toarray().T)
            n_co = b_t @ db                                               # items x block
            sim = (g / norm[:, None] / norm[None, start:stop]) * (n_co / (n_co + self.shrink))
            sim = sim.T                                                   # block x items
            sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf   # drop self
            ti, tv = _topk_rows(sim, width)
            idx[start:stop], val[start:stop] = ti, tv

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(work, range(0, n_items, block)))
        return idx, val

    def set_k(self, k: int):
        self.k = k
        v = self.nb_val[:, :k]
        rows = np.repeat(np.arange(self.n_items), k)
        keep = v.ravel() > 0
        self.sk = sp.csr_matrix((v.ravel()[keep], (rows[keep], self.nb_idx[:, :k].ravel()[keep])),
                                shape=(self.n_items, self.n_items), dtype=np.float32)
        self.sk_t = self.sk.T.tocsr()
        self.sk_abs_t = abs(self.sk).T.tocsr()
        return self

    def with_targets(self, target: sp.csr_matrix):
        """Shallow copy that scores users from `target` (users x items ratings, e.g. truncated histories)
        while keeping the item similarities learned from the full training matrix (cold-start simulation)."""
        c = copy.copy(self)
        c.mu, c.rc, c.bin = center(target)
        return c

    def _num_den(self, users: np.ndarray):
        num = (self.rc[users] @ self.sk_t).toarray()
        den = (self.bin[users] @ self.sk_abs_t).toarray()
        return num, den

    def score_fn(self, users):
        return self._num_den(users)[0]

    def predict(self, u, i):
        return _predict_pairs(self, np.asarray(u), np.asarray(i), batch=2000)


# ------------------------------------------------------------------------------ user-based
class UserKNN:
    def __init__(self, k: int = 50, shrink: float = SHRINK, batch: int = 256):
        self.k, self.shrink, self.batch = k, shrink, batch

    def fit(self, train: sp.csr_matrix):
        self.mu, self.rc, self.bin = center(train)
        self.n_users, self.n_items = train.shape
        self.rc_t = self.rc.T.tocsr()
        self.b_t = self.bin.T.tocsr()
        norm = np.sqrt(np.asarray(self.rc.multiply(self.rc).sum(axis=1)).ravel()).astype(np.float32)
        norm[norm == 0] = 1.0
        self.norm = norm
        self.t_rc, self.t_bin, self.t_norm = self.rc, self.bin, norm   # target users = the pool by default
        self.sk_cache: tuple | None = None
        return self

    def with_targets(self, target: sp.csr_matrix):
        """Shallow copy that scores users from `target` (e.g. truncated histories) while the neighbour pool
        keeps the full training histories (cold-start simulation). A user's own pool row is still excluded."""
        c = copy.copy(self)
        c.mu, c.t_rc, c.t_bin = center(target)
        n = np.sqrt(np.asarray(c.t_rc.multiply(c.t_rc).sum(axis=1)).ravel()).astype(np.float32)
        n[n == 0] = 1.0
        c.t_norm = n
        c.sk_cache = None
        return c

    def _neighbors(self, users: np.ndarray) -> sp.csr_matrix:
        key = (users[0], users[-1], len(users), self.k)
        if self.sk_cache and self.sk_cache[0] == key:
            return self.sk_cache[1]
        g = (self.t_rc[users] @ self.rc_t).toarray()
        n_co = (self.t_bin[users] @ self.b_t).toarray()
        sim = (g / self.t_norm[users][:, None] / self.norm[None, :]) * (n_co / (n_co + self.shrink))
        sim[np.arange(len(users)), users] = -np.inf                       # drop self
        ti, tv = _topk_rows(sim, self.k)
        keep = tv > 0
        rows = np.repeat(np.arange(len(users)), self.k)
        sk = sp.csr_matrix((tv.ravel()[keep.ravel()], (rows[keep.ravel()], ti.ravel()[keep.ravel()])),
                           shape=(len(users), self.n_users), dtype=np.float32)
        self.sk_cache = (key, sk)
        return sk

    def _num_den(self, users: np.ndarray):
        sk = self._neighbors(users)
        num = (sk @ self.rc).toarray()
        den = (abs(sk) @ self.bin).toarray()
        return num, den

    def score_fn(self, users):
        out = []
        for s in range(0, len(users), self.batch):
            out.append(self._num_den(users[s:s + self.batch])[0])
        return np.vstack(out)

    def predict(self, u, i):
        return _predict_pairs(self, np.asarray(u), np.asarray(i), batch=self.batch)


# ------------------------------------------------------------------------------- driver
def _setup(dataset: str):
    train, val, test = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    return train, val, test, tr, trval


def eval_model(model, dataset, split_df, exclude, n_items, rng, users=None, max_users=None):
    if users is not None:
        split_df = split_df[split_df["u"].isin(users)]
    t0 = time.time()
    r = evaluate_rating(model.predict, split_df, rng)
    t1 = time.time()
    m = evaluate_topk(model.score_fn, split_df, exclude, n_items, max_users=max_users)
    return {**r, **m, "sec_rating": round(t1 - t0, 1), "sec_topk": round(time.time() - t1, 1)}


def sample_users(n_users: int, n: int | None, seed: int = SEED):
    if n is None or n >= n_users:
        return None
    return np.sort(np.random.default_rng(seed).choice(n_users, n, replace=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(data.CONFIG), required=True)
    ap.add_argument("--model", choices=["item", "user"], required=True)
    ap.add_argument("--mode", choices=["tune", "final"], required=True)
    ap.add_argument("--users", type=int, default=None, help="random user subsample for evaluation")
    ap.add_argument("--k", type=int, default=None, help="neighbours for rating prediction (final mode)")
    ap.add_argument("--k-rank", type=int, default=None, help="neighbours for Top-K ranking (final mode)")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100, 200])
    ap.add_argument("--shrink", type=float, default=SHRINK)
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 30)

    rng = (0.5, 5.0) if a.dataset.startswith("ml") else (1.0, 5.0)
    train, val, test, tr, trval = _setup(a.dataset)
    n_users, n_items = tr.shape
    users = sample_users(n_users, a.users)
    name = f"{a.model}_knn"
    cols = ["rmse", "mae", "precision@10", "recall@10", "hit@10", "ndcg@10", "n_users", "sec_rating", "sec_topk"]

    t0 = time.time()
    if a.model == "item":
        cache = str(data.processed_dir(a.dataset) / f"itemknn_sim_shrink{int(a.shrink)}.npz")
        model = ItemKNN(k=a.ks[0], shrink=a.shrink).fit(tr, cache=cache)
        print(f"item similarity ready in {time.time()-t0:.0f}s", flush=True)
    else:
        model = UserKNN(k=a.ks[0], shrink=a.shrink).fit(tr)

    if a.mode == "tune":
        rows = []
        for k in a.ks:
            model.set_k(k) if a.model == "item" else setattr(model, "k", k)
            if a.model == "user":
                model.sk_cache = None
            m = eval_model(model, a.dataset, val, tr, n_items, rng, users, max_users=a.users)
            params = {"k": k, "shrink": a.shrink, "eval_users": a.users or "all"}
            log_result(a.dataset, name, "val", params, m, notes="tuning")
            rows.append({"k": k, **m})
            print(pd.DataFrame(rows)[["k"] + cols].round(4).to_string(index=False), flush=True)
    else:
        # k chosen on validation separately for rating prediction (k) and Top-K ranking (k_rank)
        k_rank = a.k_rank or a.k

        def set_k(kk):
            model.set_k(kk) if a.model == "item" else setattr(model, "k", kk)
            if a.model == "user":
                model.sk_cache = None

        test_u = test if users is None else test[test["u"].isin(users)]
        set_k(a.k)
        r = evaluate_rating(model.predict, test_u, rng)
        set_k(k_rank)
        t = evaluate_topk(model.score_fn, test_u, trval, n_items, max_users=a.users)
        m = {**r, **t}
        log_result(a.dataset, name, "test",
                   {"k_rating": a.k, "k_rank": k_rank, "shrink": a.shrink, "eval_users": a.users or "all"},
                   m, notes="final")
        print(pd.Series(m).round(4).to_string())
