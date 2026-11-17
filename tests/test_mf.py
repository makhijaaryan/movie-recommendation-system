import numpy as np
import pandas as pd

from src.mf import MF


def synthetic(n_users=300, n_items=200, k=4, density=0.25, noise=0.05, seed=0, offset=3.0):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(0, 0.8, (n_users, k)), rng.normal(0, 0.8, (n_items, k))
    bu, bi = rng.normal(0, 0.3, n_users), rng.normal(0, 0.3, n_items)
    full = offset + bu[:, None] + bi[None, :] + P @ Q.T + rng.normal(0, noise, (n_users, n_items))
    mask = rng.random((n_users, n_items)) < density
    u, i = np.nonzero(mask)
    df = pd.DataFrame({"u": u.astype(np.int32), "i": i.astype(np.int32), "rating": full[u, i].astype(np.float32)})
    df["ts"] = 0
    msk = rng.random(len(df)) < 0.85
    return df[msk].reset_index(drop=True), df[~msk].reset_index(drop=True), n_users, n_items


def test_biased_mf_learns_structure_and_beats_global_mean():
    tr, va, nu, ni = synthetic()
    m = MF(k=4, lr=0.05, lam=0.02, biased=True, init_std=0.3).fit(tr, nu, ni, va, max_epochs=100, patience=40, verbose=False)
    rmse = np.sqrt(np.mean((va.rating - m.predict(va.u, va.i)) ** 2))
    base = np.sqrt(np.mean((va.rating - tr.rating.mean()) ** 2))
    assert rmse < 0.5 * base


def test_plain_mf_also_learns():
    tr, va, nu, ni = synthetic()
    m = MF(k=6, lr=0.05, lam=0.02, biased=False, init_std=0.3).fit(tr, nu, ni, va, max_epochs=100, patience=40, verbose=False)
    rmse = np.sqrt(np.mean((va.rating - m.predict(va.u, va.i)) ** 2))
    base = np.sqrt(np.mean((va.rating - tr.rating.mean()) ** 2))
    assert rmse < 0.8 * base


def test_training_is_deterministic_given_seed():
    tr, va, nu, ni = synthetic(n_users=80, n_items=60)
    a = MF(k=3, lr=0.02, lam=0.02).fit(tr, nu, ni, va, max_epochs=5, verbose=False, seed=1)
    b = MF(k=3, lr=0.02, lam=0.02).fit(tr, nu, ni, va, max_epochs=5, verbose=False, seed=1)
    c = MF(k=3, lr=0.02, lam=0.02).fit(tr, nu, ni, va, max_epochs=5, verbose=False, seed=2)
    assert np.array_equal(a.P, b.P) and not np.array_equal(a.P, c.P)


def test_early_stopping_restores_best_weights():
    tr, va, nu, ni = synthetic(n_users=100, n_items=80)
    m = MF(k=8, lr=0.05, lam=0.0, biased=True).fit(tr, nu, ni, va, max_epochs=40, patience=2, verbose=False)
    rmse = np.sqrt(np.mean((va.rating - m.predict(va.u, va.i)) ** 2))
    assert abs(rmse - m.best_val_rmse) < 1e-3          # kept weights == best epoch's weights


def test_score_fn_shape_and_item_bias_used():
    tr, va, nu, ni = synthetic(n_users=50, n_items=40)
    m = MF(k=3, lr=0.02, lam=0.02).fit(tr, nu, ni, va, max_epochs=3, verbose=False)
    s = m.score_fn(np.arange(5))
    assert s.shape == (5, ni) and np.all(np.isfinite(s))
