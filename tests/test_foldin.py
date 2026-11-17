import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.cf import ItemKNN, UserKNN
from src.foldin import foldin_mf, foldin_occf
from src.mf import MF
from src.occf import OCCF


def lowrank_ratings(n_users=120, n_items=60, k=3, seed=0, density=0.5):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
    R = np.clip(3.0 + P @ Q.T, 0.5, 5.0)
    mask = rng.random((n_users, n_items)) < density
    u, i = np.nonzero(mask)
    return pd.DataFrame({"u": u.astype(np.int32), "i": i.astype(np.int32), "rating": R[u, i].astype(np.float32)}), n_users, n_items


def to_csr(df, n_users, n_items):
    return sp.csr_matrix((df.rating.to_numpy(np.float32), (df.u, df.i)), shape=(n_users, n_items))


def test_occf_foldin_of_training_users_reproduces_their_vectors():
    df, nu, ni = lowrank_ratings()
    R = to_csr(df, nu, ni)
    pos = R.copy()
    pos.data = (pos.data >= 4.0).astype(np.float32)
    pos.eliminate_zeros()
    m = OCCF(k=4, alpha=5.0, lam=1.0, iters=15, init_std=0.1).fit(pos, verbose=False)
    users = np.arange(nu)
    P = foldin_occf(m, R, users)
    # fold-in is the exact ALS user step, so after convergence it matches the trained user vectors
    cos = (P * m.P).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(m.P, axis=1) + 1e-12)
    assert np.median(cos) > 0.98


def test_occf_foldin_user_without_positives_gets_zero_vector():
    df, nu, ni = lowrank_ratings()
    pos = to_csr(df, nu, ni)
    pos.data[:] = 1.0
    m = OCCF(k=4, alpha=5.0, lam=1.0, iters=3, init_std=0.1).fit(pos, verbose=False)
    low = sp.csr_matrix(([2.0, 3.0], ([0, 0], [1, 2])), shape=(nu, ni))      # only ratings below 4
    assert not foldin_occf(m, low, [0]).any()


def test_mf_foldin_predicts_training_users_ratings():
    df, nu, ni = lowrank_ratings(n_users=200, n_items=50, density=0.6)
    for biased in (False, True):
        m = MF(k=4, lr=0.02, lam=0.002, biased=biased, lr_decay=0.98, init_std=0.3).fit(
            df, nu, ni, None, fixed_epochs=60, verbose=False)
        P, bu = foldin_mf(m, to_csr(df, nu, ni), np.arange(nu))
        pred_model = m.predict(df.u.to_numpy(), df.i.to_numpy())
        pred_fold = np.einsum("nf,nf->n", P[df.u], m.Q[df.i]) + (m.mu + bu[df.u] + m.bi[df.i] if biased else 0)
        assert np.corrcoef(pred_model, pred_fold)[0, 1] > 0.97
        assert np.sqrt(np.mean((pred_fold - df.rating) ** 2)) < 0.6


def test_mf_foldin_with_few_ratings_shrinks_toward_zero():
    df, nu, ni = lowrank_ratings(n_users=200, n_items=50, density=0.6)
    m = MF(k=4, lr=0.02, lam=0.05, biased=True, lr_decay=0.98, init_std=0.3).fit(df, nu, ni, None, fixed_epochs=20,
                                                                                verbose=False)
    one = sp.csr_matrix(([5.0], ([0], [3])), shape=(nu, ni))
    P, bu = foldin_mf(m, one, [0, 1])                      # user 1 has no ratings
    assert not P[1].any() and bu[1] == 0
    assert np.linalg.norm(P[0]) < 2.0                      # strong ridge keeps a 1-rating vector small


def test_cf_with_targets_identity_and_isolation():
    df, nu, ni = lowrank_ratings(n_users=80, n_items=40, density=0.4)
    R = to_csr(df, nu, ni)
    for model in (ItemKNN(k=10).fit(R), UserKNN(k=10).fit(R)):
        users = np.arange(20)
        base = model.score_fn(users)
        same = model.with_targets(R).score_fn(users)
        assert np.allclose(base, same, atol=1e-5)          # same matrix -> identical scores
        sparse = R.copy().tolil()
        sparse[0, :] = 0                                    # empty out user 0's history only
        sparse = sparse.tocsr()
        sparse.eliminate_zeros()
        cut = model.with_targets(sparse).score_fn(users)
        assert not np.any(cut[0])                           # nothing known about user 0 -> zero scores
        assert np.allclose(cut[1:], base[1:], atol=1e-5)    # other users unaffected
