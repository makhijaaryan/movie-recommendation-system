"""Matrix factorization trained with SGD (proposal §4.4).

Plain MF (the proposal's model):   r_hat(u,i) = p_u . q_i
Biased MF (extension, D21):        r_hat(u,i) = mu + b_u + b_i + p_u . q_i

Objective over observed ratings Omega (per-sample SGD form of the proposal's loss):
    sum (r_ui - r_hat_ui)^2 + lambda * (|p_u|^2 + |q_i|^2 [+ b_u^2 + b_i^2])
Update:  e = r - r_hat;  p += lr*(e*q - lam*p);  q += lr*(e*p_old - lam*q);  b += lr*(e - lam*b)

Learning rate decays geometrically per epoch; training stops early when validation
RMSE has not improved for `patience` epochs and the best-epoch weights are kept.
The SGD kernel is compiled with Numba and is single-threaded, so runs are
deterministic given the seed.

Run from the project root, e.g.
    python -m src.mf --dataset ml_small --mode tune --biased 1 --k 32 --lr 0.01 --lam 0.05
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

from src import data
from src.eval import evaluate_rating, evaluate_topk, log_result


@njit(fastmath=True, cache=True)
def _sgd_epoch(u, i, r, order, P, Q, bu, bi, mu, lr, lam, biased):
    k = P.shape[1]
    for t in range(order.shape[0]):
        idx = order[t]
        uu = u[idx]
        ii = i[idx]
        dot = np.float32(0.0)
        for f in range(k):
            dot += P[uu, f] * Q[ii, f]
        pred = dot
        if biased:
            pred += mu + bu[uu] + bi[ii]
        e = r[idx] - pred
        if biased:
            bu[uu] += lr * (e - lam * bu[uu])
            bi[ii] += lr * (e - lam * bi[ii])
        for f in range(k):
            pf = P[uu, f]
            qf = Q[ii, f]
            P[uu, f] = pf + lr * (e * qf - lam * pf)
            Q[ii, f] = qf + lr * (e * pf - lam * qf)


class MF:
    def __init__(self, k=32, lr=0.01, lam=0.05, biased=True, lr_decay=0.95, init_std=0.05):
        self.k, self.lr, self.lam, self.biased = k, lr, lam, biased
        self.lr_decay, self.init_std = lr_decay, init_std

    # ------------------------------------------------------------------ training
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, val: pd.DataFrame | None = None,
            max_epochs: int = 40, patience: int = 2, seed: int = 42, verbose: bool = True,
            fixed_epochs: int | None = None):
        rng = np.random.default_rng(seed)
        u = train["u"].to_numpy(np.int32)
        i = train["i"].to_numpy(np.int32)
        r = train["rating"].to_numpy(np.float32)
        self.mu = np.float32(r.mean()) if self.biased else np.float32(0.0)
        # Plain MF has no mean term, so start p.q at the mean rating: all entries ~ sqrt(mean/k).
        base = 0.0 if self.biased else np.sqrt(r.mean() / self.k)
        self.P = (base + self.init_std * rng.standard_normal((n_users, self.k))).astype(np.float32)
        self.Q = (base + self.init_std * rng.standard_normal((n_items, self.k))).astype(np.float32)
        self.bu = np.zeros(n_users, np.float32)
        self.bi = np.zeros(n_items, np.float32)
        best = (np.inf, 0, None)
        self.history = []
        epochs = fixed_epochs or max_epochs
        for ep in range(1, epochs + 1):
            lr = np.float32(self.lr * self.lr_decay ** (ep - 1))
            order = rng.permutation(len(u)).astype(np.int64)
            t0 = time.time()
            _sgd_epoch(u, i, r, order, self.P, self.Q, self.bu, self.bi, self.mu, lr,
                       np.float32(self.lam), self.biased)
            row = {"epoch": ep, "lr": float(lr), "sec": round(time.time() - t0, 1)}
            tr_rmse = float(np.sqrt(np.mean((r[:2_000_000] - self.predict(u[:2_000_000], i[:2_000_000])) ** 2)))
            row["train_rmse"] = round(tr_rmse, 4)
            if val is not None:
                vr = float(np.sqrt(np.mean((val["rating"].to_numpy() - self.predict(val["u"].to_numpy(),
                                                                                   val["i"].to_numpy())) ** 2)))
                row["val_rmse"] = round(vr, 4)
                if vr < best[0] - 1e-5:
                    best = (vr, ep, (self.P.copy(), self.Q.copy(), self.bu.copy(), self.bi.copy()))
                elif fixed_epochs is None and ep - best[1] >= patience:
                    self.history.append(row)
                    if verbose:
                        print(row, flush=True)
                    break
            self.history.append(row)
            if verbose:
                print(row, flush=True)
            if not np.isfinite(row["train_rmse"]):
                break                                  # diverged
        if val is not None and best[2] is not None and fixed_epochs is None:
            self.P, self.Q, self.bu, self.bi = best[2]
        self.best_epoch, self.best_val_rmse = best[1], best[0]
        return self

    # ---------------------------------------------------------------- prediction
    def predict(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        out = np.einsum("nf,nf->n", self.P[u], self.Q[i])
        if self.biased:
            out = out + self.mu + self.bu[u] + self.bi[i]
        return out

    def score_fn(self, users):
        s = self.P[users] @ self.Q.T
        if self.biased:
            s = s + self.bi[None, :]              # user terms are constant per row; irrelevant for ranking
        return s



# ------------------------------------------------------------------------------ grid search
def run_config(dataset, train, val, tr, n_users, n_items, rng, k, lr, lam, biased, decay=0.97,
               max_epochs=60, patience=3, rank_users=20000, tag="grid"):
    """Train one configuration with early stopping, evaluate on validation, log, return metrics."""
    model = MF(k, lr, lam, biased, decay).fit(train, n_users, n_items, val, max_epochs=max_epochs,
                                              patience=patience, verbose=False)
    r = evaluate_rating(model.predict, val, rng)
    t = evaluate_topk(model.score_fn, val, tr, n_items, max_users=rank_users)
    params = {"k": k, "lr": lr, "lam": lam, "decay": decay, "biased": biased,
              "best_epoch": model.best_epoch, "eval_users": rank_users}
    name = "mf_biased" if biased else "mf_plain"
    log_result(dataset, name, "val", params, {**r, **t}, notes=f"tuning {tag}")
    out = {"k": k, "lr": lr, "lam": lam, "best_epoch": model.best_epoch, **r, **t}
    print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out.items()
           if kk in ("k", "lr", "lam", "best_epoch", "rmse", "ndcg@10", "hit@10")}, flush=True)
    return out


def grid_search(dataset, biased):
    """Staged search (each stage keeps the best-validation-RMSE value so far):
       1. lr x lam at k=32;  2. k in {8,16,32,64,128};  3. refine lam (x0.5, x0.75, x1.5, x2) at best k."""
    rng = (0.5, 5.0) if dataset.startswith("ml") else (1.0, 5.0)
    train, val, test, tr, trval = _setup(dataset)
    n_users, n_items = tr.shape
    runs = []

    def go(k, lr, lam, tag):
        key = (k, lr, round(lam, 5))
        if any((x["k"], x["lr"], round(x["lam"], 5)) == key for x in runs):
            return
        runs.append(run_config(dataset, train, val, tr, n_users, n_items, rng, k, lr, lam, biased, tag=tag))

    best = lambda: min(runs, key=lambda x: x["rmse"])
    for lr in (0.005, 0.01, 0.02):
        for lam in (0.02, 0.05, 0.1):
            go(32, lr, lam, "stage1")
    b = best()
    for k in (8, 16, 64, 128):
        go(k, b["lr"], b["lam"], "stage2")
    b = best()
    for f in (0.5, 0.75, 1.5, 2.0):
        go(b["k"], b["lr"], b["lam"] * f, "stage3")
    df = pd.DataFrame(runs)
    print("\nBEST by val RMSE   :", df.loc[df.rmse.idxmin(), ["k", "lr", "lam", "best_epoch", "rmse", "ndcg@10"]].to_dict())
    print("BEST by val NDCG@10:", df.loc[df["ndcg@10"].idxmax(), ["k", "lr", "lam", "best_epoch", "rmse", "ndcg@10"]].to_dict())
    return df

# ------------------------------------------------------------------------------ driver
def _setup(dataset):
    train, val, test = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    return train, val, test, tr, trval


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(data.CONFIG), required=True)
    ap.add_argument("--mode", choices=["tune", "grid", "final"], required=True)
    ap.add_argument("--biased", type=int, default=1)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=None, help="fixed epochs (final mode: best epoch from tuning)")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--rank-users", type=int, default=20000, help="val users for ranking metrics in tuning")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    if a.mode == "grid":
        grid_search(a.dataset, bool(a.biased))
        raise SystemExit
    rng = (0.5, 5.0) if a.dataset.startswith("ml") else (1.0, 5.0)
    train, val, test, tr, trval = _setup(a.dataset)
    n_users, n_items = tr.shape
    name = "mf_biased" if a.biased else "mf_plain"
    params = {"k": a.k, "lr": a.lr, "lam": a.lam, "decay": a.decay, "biased": bool(a.biased)}

    model = MF(a.k, a.lr, a.lam, bool(a.biased), a.decay)
    if a.mode == "tune":
        model.fit(train, n_users, n_items, val, max_epochs=a.max_epochs)
        r = evaluate_rating(model.predict, val, rng)
        t = evaluate_topk(model.score_fn, val, tr, n_items, max_users=a.rank_users)
        params |= {"best_epoch": model.best_epoch, "eval_users": a.rank_users}
        log_result(a.dataset, name, "val", params, {**r, **t}, notes="tuning " + a.tag)
        print(pd.Series({**r, **t}).round(4).to_string())
    else:
        # retrain on train only for the best number of epochs found on validation, then test
        model.fit(train, n_users, n_items, val, fixed_epochs=a.epochs)
        r = evaluate_rating(model.predict, test, rng)
        t = evaluate_topk(model.score_fn, test, trval, n_items)
        params |= {"epochs": a.epochs}
        log_result(a.dataset, name, "test", params, {**r, **t}, notes="final " + a.tag)
        print(pd.Series({**r, **t}).round(4).to_string())
