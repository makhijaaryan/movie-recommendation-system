"""Fold-in: compute a user's latent vector from a (short) rating history WITHOUT retraining, keeping the
item factors fixed. Used by the simulated cold-start experiment (src/coldstart.py) and, later, by a demo
that takes a new user's ratings.

* One-class CF (WRMF): the exact closed-form ALS user step. With positives = ratings >= threshold,
      p = (QtQ + lam*I + alpha * sum_pos q q^T)^-1 * (1 + alpha) * sum_pos q
  A user with no positive rating gets p = 0 (no signal).
  For the dual-regularized model the user-graph term is dropped (a new user has no graph neighbours);
  the item graph already shaped Q during training.
* Matrix factorization (SGD, src/mf.py): the SGD kernel regularizes per rating, i.e. minimises
      sum_{i in R_u} (r_ui - mu - b_u - b_i - p.q_i)^2 + lam * |R_u| * (|p|^2 + b_u^2)
  over (p, b_u) (b_u only for the biased model), a ridge regression with strength lam*|R_u|.

Both take a CSR matrix (users x items) of the ratings to fold in and return vectors for the users asked for.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.eval import RELEVANT_THRESHOLD


def foldin_occf(model, ratings: sp.csr_matrix, users, threshold: float = RELEVANT_THRESHOLD) -> np.ndarray:
    """Latent vectors (len(users), k) of a fitted OCCF for `users`, from their rows of `ratings`."""
    Q = model.Q.astype(np.float64)
    k = Q.shape[1]
    base = Q.T @ Q + model.lam * np.eye(k)
    out = np.zeros((len(users), k), dtype=np.float32)
    ratings = ratings.tocsr()
    for n, u in enumerate(users):
        sl = slice(ratings.indptr[u], ratings.indptr[u + 1])
        idx, r = ratings.indices[sl], ratings.data[sl]
        pos = idx[r >= threshold]
        if len(pos) == 0:
            continue
        Qp = Q[pos]
        A = base + model.alpha * (Qp.T @ Qp)
        b = (1.0 + model.alpha) * Qp.sum(axis=0)
        out[n] = np.linalg.solve(A, b)
    return out


def foldin_mf(model, ratings: sp.csr_matrix, users) -> tuple[np.ndarray, np.ndarray]:
    """(P, bu) for `users` from their rows of `ratings`; bu is all zeros for the plain model."""
    Q = model.Q.astype(np.float64)
    k = Q.shape[1]
    P = np.zeros((len(users), k), dtype=np.float32)
    bu = np.zeros(len(users), dtype=np.float32)
    ratings = ratings.tocsr()
    for n, u in enumerate(users):
        sl = slice(ratings.indptr[u], ratings.indptr[u + 1])
        idx, r = ratings.indices[sl], ratings.data[sl].astype(np.float64)
        m = len(idx)
        if m == 0:
            continue
        reg = model.lam * m
        if model.biased:
            A = np.hstack([Q[idx], np.ones((m, 1))])
            y = r - float(model.mu) - model.bi[idx]
        else:
            A = Q[idx]
            y = r
        theta = np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ y)
        P[n] = theta[:k]
        if model.biased:
            bu[n] = theta[k]
    return P, bu
