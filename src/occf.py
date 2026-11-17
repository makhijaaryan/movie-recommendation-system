"""One-class collaborative filtering (proposal §4.5, after Yao et al., CIKM 2014).

Setting: y_ui = 1 if the user rated the item >= 4 in the training split, else 0
(unobserved and low ratings are both "not known positive"; they are not treated as
true dislikes but get a small weight through the confidence scheme below).

Model (WRMF, weighted matrix factorization solved by ALS):
    min  sum_{u,i} c_ui (y_ui - p_u . q_i)^2 + lam (|P|^2 + |Q|^2)
         + lam_u * sum_{u,v} s_uv |p_u - p_v|^2  +  lam_i * sum_{i,j} s_ij |q_i - q_j|^2
with c_ui = 1 + alpha for positives and 1 otherwise.  The last two terms are the
"dual" graph regularisers of Yao et al.: users (items) that are similar in their
positive interactions are pulled together in latent space.  lam_u = lam_i = 0 gives
plain WRMF.  Similarities are cosine similarity of binary positive-interaction
vectors, top-m neighbours per user/item.

Each ALS half-step solves, per user (item), a k x k system in closed form:
    (Y^T Y + alpha * Y_pos^T Y_pos + (lam + lam_g * d) I) x = (1+alpha) * sum_pos y + lam_g * sum_v s_v x_v
(graph neighbours use the previous iteration's factors, so the result is deterministic).

Run from the project root, e.g.
    python -m src.occf --dataset ml_small --mode tune --k 32 --alpha 20 --lam 10
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange

from src import data
from src.eval import RELEVANT_THRESHOLD, evaluate_topk, log_result


# ------------------------------------------------------------------------------ data
def positives_csr(train: pd.DataFrame, n_users: int, n_items: int, thr: float = RELEVANT_THRESHOLD):
    """Binary users x items matrix of training positives (rating >= thr)."""
    pos = train[train["rating"] >= thr]
    return sp.csr_matrix((np.ones(len(pos), dtype=np.float32), (pos["u"], pos["i"])),
                         shape=(n_users, n_items))


def knn_graph(m: sp.csr_matrix, top_m: int = 20, threads: int = 4, block: int = 256):
    """Cosine-similarity kNN graph among the ROWS of binary matrix m (n x d).
    Returns (idx int32 (n, top_m), w float32 (n, top_m)); self excluded; weights >= 0."""
    n = m.shape[0]
    top_m = min(top_m, n - 1)
    m = m.tocsr().astype(np.float32)
    mt = m.T.tocsr()
    norm = np.sqrt(np.asarray(m.sum(axis=1)).ravel()).astype(np.float32)
    norm[norm == 0] = 1.0
    idx = np.zeros((n, top_m), np.int32)
    w = np.zeros((n, top_m), np.float32)

    def work(start):
        stop = min(start + block, n)
        g = (m[start:stop] @ mt).toarray()
        g = g / norm[start:stop, None] / norm[None, :]
        g[np.arange(stop - start), np.arange(start, stop)] = -1.0
        part = np.argpartition(-g, top_m - 1, axis=1)[:, :top_m]
        vals = np.take_along_axis(g, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        w[start:stop] = np.maximum(np.take_along_axis(vals, order, axis=1), 0.0)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(0, n, block)))
    return idx, w


# ------------------------------------------------------------------------------ ALS kernel
@njit(parallel=True, fastmath=True, cache=True)
def _half_step(indptr, indices, Y, YtY, alpha, lam, use_g, g_idx, g_w, g_sum, cur, lam_g, out):
    n, k = out.shape
    for u in prange(n):
        A = YtY + lam * np.eye(k)
        b = np.zeros(k)
        for p in range(indptr[u], indptr[u + 1]):
            j = indices[p]
            for a in range(k):
                ya = np.float64(Y[j, a])
                b[a] += (1.0 + alpha) * ya
                for c in range(k):
                    A[a, c] += alpha * ya * np.float64(Y[j, c])
        if use_g:
            for a in range(k):
                A[a, a] += lam_g * g_sum[u]
            for t in range(g_idx.shape[1]):
                v = g_idx[u, t]
                wv = g_w[u, t]
                for a in range(k):
                    b[a] += lam_g * wv * np.float64(cur[v, a])
        x = np.linalg.solve(A, b)
        for a in range(k):
            out[u, a] = x[a]


class OCCF:
    def __init__(self, k=64, alpha=20.0, lam=10.0, lam_u=0.0, lam_i=0.0, iters=15, init_std=0.01):
        self.k, self.alpha, self.lam = k, alpha, lam
        self.lam_u, self.lam_i, self.iters, self.init_std = lam_u, lam_i, iters, init_std

    def fit(self, pos: sp.csr_matrix, user_graph=None, item_graph=None, seed=42, callback=None,
            verbose=True):
        """callback(model, it) -> float score (higher is better) evaluated each iteration; the best
        iteration's factors are kept and self.best_iter is set."""
        rng = np.random.default_rng(seed)
        n_users, n_items = pos.shape
        pos = pos.tocsr()
        pos_t = pos.T.tocsr()
        self.P = (self.init_std * rng.standard_normal((n_users, self.k))).astype(np.float32)
        self.Q = (self.init_std * rng.standard_normal((n_items, self.k))).astype(np.float32)
        ug = ig = None
        if user_graph is not None and self.lam_u > 0:
            ug = (user_graph[0].astype(np.int32), user_graph[1].astype(np.float32),
                  user_graph[1].sum(axis=1).astype(np.float64))
        if item_graph is not None and self.lam_i > 0:
            ig = (item_graph[0].astype(np.int32), item_graph[1].astype(np.float32),
                  item_graph[1].sum(axis=1).astype(np.float64))
        dummy_i, dummy_w, dummy_s = np.zeros((1, 1), np.int32), np.zeros((1, 1), np.float32), np.zeros(1)
        best = (-np.inf, 0, None)
        self.history = []
        for it in range(1, self.iters + 1):
            t0 = time.time()
            qtq = (self.Q.T.astype(np.float64) @ self.Q.astype(np.float64))
            p_old = self.P.copy()
            gi, gw, gs = ug if ug else (dummy_i, dummy_w, dummy_s)
            _half_step(pos.indptr, pos.indices, self.Q, qtq, self.alpha, self.lam, ug is not None,
                       gi, gw, gs, p_old, self.lam_u, self.P)
            ptp = (self.P.T.astype(np.float64) @ self.P.astype(np.float64))
            q_old = self.Q.copy()
            gi, gw, gs = ig if ig else (dummy_i, dummy_w, dummy_s)
            _half_step(pos_t.indptr, pos_t.indices, self.P, ptp, self.alpha, self.lam, ig is not None,
                       gi, gw, gs, q_old, self.lam_i, self.Q)
            row = {"iter": it, "sec": round(time.time() - t0, 1)}
            if callback is not None:
                row["score"] = float(callback(self, it))
                if row["score"] > best[0] + 1e-6:
                    best = (row["score"], it, (self.P.copy(), self.Q.copy()))
            self.history.append(row)
            if verbose:
                print(row, flush=True)
        if best[2] is not None:
            self.P, self.Q = best[2]
        self.best_iter, self.best_score = best[1], best[0]
        return self

    def score_fn(self, users):
        return self.P[users] @ self.Q.T


# ------------------------------------------------------------------------------ driver
def _setup(dataset):
    train, val, test = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    return train, val, test, tr, trval


def get_graphs(dataset, pos, top_m, need_user, need_item):
    """Build (or load cached) similarity graphs from the training positives."""
    out = {}
    for name, needed, mat in (("user", need_user, pos), ("item", need_item, pos.T.tocsr())):
        if not needed:
            out[name] = None
            continue
        f = data.PROC / dataset / f"occf_graph_{name}_m{top_m}.npz"
        if f.exists():
            z = np.load(f)
            out[name] = (z["idx"], z["w"])
        else:
            t0 = time.time()
            idx, w = knn_graph(mat, top_m)
            np.savez(f, idx=idx, w=w)
            print(f"built {name} graph in {time.time()-t0:.0f}s", flush=True)
            out[name] = (idx, w)
    return out


def run_config(dataset, train, val, tr, pos, graphs, k, alpha, lam, lam_u, lam_i, iters=15, top_m=20,
               rank_users=20000, tag="grid", verbose=False):
    n_users, n_items = tr.shape
    model = OCCF(k, alpha, lam, lam_u, lam_i, iters)
    cb = lambda m, it: evaluate_topk(m.score_fn, val, tr, n_items, max_users=rank_users // 2)["ndcg@10"]
    model.fit(pos, graphs.get("user"), graphs.get("item"), callback=cb, verbose=verbose)
    t = evaluate_topk(model.score_fn, val, tr, n_items, max_users=rank_users)
    params = {"k": k, "alpha": alpha, "lam": lam, "lam_u": lam_u, "lam_i": lam_i, "iters": iters,
              "best_iter": model.best_iter, "top_m": top_m, "eval_users": rank_users}
    name = "occf_dual" if (lam_u > 0 or lam_i > 0) else "occf_wrmf"
    log_result(dataset, name, "val", params, t, notes=f"tuning {tag}")
    print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in
           {"k": k, "alpha": alpha, "lam": lam, "lam_u": lam_u, "lam_i": lam_i,
            "best_iter": model.best_iter, "ndcg@10": t["ndcg@10"], "hit@10": t["hit@10"]}.items()}, flush=True)
    return {"k": k, "alpha": alpha, "lam": lam, "lam_u": lam_u, "lam_i": lam_i,
            "best_iter": model.best_iter, **t}


def grid_search(dataset, dual: bool, top_m=20):
    """Staged search on validation NDCG@10 (each stage keeps the best so far).
    WRMF:  1. alpha x lam at k=64;  2. k in {32,128}.
    Dual:  starts from the best WRMF config; 3. lam_u / lam_i in {0.1, 1, 10} (item-only, user-only, both)."""
    train, val, test, tr, trval = _setup(dataset)
    n_users, n_items = tr.shape
    pos = positives_csr(train, n_users, n_items)
    runs = []

    def go(**kw):
        key = tuple(sorted(kw.items()))
        if any(r.get("_key") == key for r in runs):
            return
        res = run_config(dataset, train, val, tr, pos, G, top_m=top_m, tag=kw.pop("tag", "grid"), **{k: v for k, v in kw.items()})
        res["_key"] = key
        runs.append(res)

    best = lambda: max(runs, key=lambda r: r["ndcg@10"])
    G = {"user": None, "item": None}
    if not dual:
        for alpha in (5, 20, 60):
            for lam in (1, 10, 50):
                go(k=64, alpha=alpha, lam=lam, lam_u=0.0, lam_i=0.0, tag="stage1")
        b = best()
        for k in (32, 128):
            go(k=k, alpha=b["alpha"], lam=b["lam"], lam_u=0.0, lam_i=0.0, tag="stage2")
    else:
        G = get_graphs(dataset, pos, top_m, True, True)
        w = pd.read_csv(RESULTS)  # base WRMF config = best WRMF val NDCG so far for this dataset
        w = w[(w.dataset == dataset) & (w.model == "occf_wrmf") & (w.split == "val")]
        import json
        pr = w.params.apply(json.loads).apply(pd.Series)
        w = pd.concat([w.reset_index(drop=True), pr.reset_index(drop=True)], axis=1)
        b = w.loc[w["ndcg@10"].idxmax()]
        base = dict(k=int(b.k), alpha=float(b.alpha), lam=float(b.lam))
        print("base WRMF config:", base, flush=True)
        for lu, li in ((0, 1), (0, 10), (1, 0), (10, 0), (1, 1), (10, 10)):
            go(**base, lam_u=float(lu), lam_i=float(li), tag="dual")
    df = pd.DataFrame(runs).drop(columns="_key")
    print("\nBEST by val NDCG@10:", df.loc[df["ndcg@10"].idxmax(), ["k", "alpha", "lam", "lam_u", "lam_i", "best_iter", "ndcg@10"]].to_dict())
    return df


RESULTS = data.ROOT / "experiments" / "results.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(data.CONFIG), required=True)
    ap.add_argument("--mode", choices=["tune", "grid", "final"], required=True)
    ap.add_argument("--dual", type=int, default=0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--lam-u", type=float, default=0.0)
    ap.add_argument("--lam-i", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--top-m", type=int, default=20)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    pd.set_option("display.width", 250)

    if a.mode == "grid":
        grid_search(a.dataset, bool(a.dual), a.top_m)
        raise SystemExit
    train, val, test, tr, trval = _setup(a.dataset)
    n_users, n_items = tr.shape
    pos = positives_csr(train, n_users, n_items)
    G = get_graphs(a.dataset, pos, a.top_m, a.lam_u > 0, a.lam_i > 0)
    if a.mode == "tune":
        run_config(a.dataset, train, val, tr, pos, G, a.k, a.alpha, a.lam, a.lam_u, a.lam_i, a.iters,
                   a.top_m, tag=a.tag or "manual", verbose=True)
    else:
        # retrain on train positives for the best iteration count found on validation, then test once
        model = OCCF(a.k, a.alpha, a.lam, a.lam_u, a.lam_i, a.iters).fit(
            pos, G.get("user"), G.get("item"), verbose=False)
        t = evaluate_topk(model.score_fn, test, trval, n_items)
        name = "occf_dual" if (a.lam_u > 0 or a.lam_i > 0) else "occf_wrmf"
        log_result(a.dataset, name, "test",
                   {"k": a.k, "alpha": a.alpha, "lam": a.lam, "lam_u": a.lam_u, "lam_i": a.lam_i,
                    "iters": a.iters, "top_m": a.top_m}, t, notes="final " + a.tag)
        print(pd.Series(t).round(4).to_string())
