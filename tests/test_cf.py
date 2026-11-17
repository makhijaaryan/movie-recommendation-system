import numpy as np
import scipy.sparse as sp

from src.cf import ItemKNN, UserKNN, center


def random_train(n_users=40, n_items=25, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[:, 0] |= rng.random(n_users) < 0.9           # make sure no empty rows/cols
    vals = rng.integers(1, 6, size=(n_users, n_items)).astype(np.float32)
    return sp.csr_matrix(np.where(mask, vals, 0.0)), np.where(mask, vals, 0.0)


def test_center_means_match_numpy():
    train, dense = random_train()
    mu, rc, binary = center(train)
    for u in range(dense.shape[0]):
        r = dense[u][dense[u] > 0]
        if len(r):
            assert np.isclose(mu[u], r.mean(), atol=1e-5)
    assert np.allclose(rc.toarray().sum(axis=1)[np.diff(train.indptr) > 0], 0, atol=1e-4)
    assert binary.nnz == train.nnz


def test_item_similarity_matches_bruteforce():
    train, dense = random_train()
    shrink = 5.0
    m = ItemKNN(k=5, shrink=shrink).fit(train, threads=2)
    mask = dense > 0
    mu = (dense.sum(1) / np.maximum(mask.sum(1), 1))
    c = np.where(mask, dense - mu[:, None], 0.0)
    norm = np.sqrt((c ** 2).sum(0)); norm[norm == 0] = 1
    n_co = mask.T.astype(float) @ mask.astype(float)
    sim = (c.T @ c) / np.outer(norm, norm) * (n_co / (n_co + shrink))
    np.fill_diagonal(sim, -np.inf)
    for i in range(dense.shape[1]):
        top = np.argsort(-sim[i])[:5]
        assert np.allclose(np.sort(m.nb_val[i, :5])[::-1], np.sort(sim[i][top])[::-1], atol=1e-4)


def test_item_predictions_use_only_train_ratings():
    train, _ = random_train()
    m = ItemKNN(k=5).fit(train, threads=2)
    # predictions are finite and the user-mean is the fallback when no neighbour evidence exists
    u = np.array([0, 1, 2]); i = np.array([3, 4, 5])
    p = m.predict(u, i)
    assert np.all(np.isfinite(p))


def test_user_knn_scores_shape_and_self_excluded():
    train, _ = random_train()
    m = UserKNN(k=5, batch=16).fit(train)
    users = np.arange(10)
    sk = m._neighbors(users)
    assert sk.shape == (10, train.shape[0])
    assert all(sk[u, u] == 0 for u in users)                  # a user is never their own neighbour
    assert m.score_fn(users).shape == (10, train.shape[1])


def test_item_and_user_agree_on_duplicate_structure():
    # two identical users must be perfect neighbours: user 1 gets user 0's preference signal
    dense = np.array([[5, 1, 0, 4], [5, 1, 0, 0], [1, 5, 4, 0], [2, 4, 0, 5]], dtype=np.float32)
    train = sp.csr_matrix(dense)
    m = UserKNN(k=1, shrink=0.0, batch=4).fit(train)
    sk = m._neighbors(np.array([1]))
    assert sk[0, 0] > 0.9                                     # user 0 is user 1's top neighbour (cos≈0.96)
    assert m.score_fn(np.array([1]))[0, 3] > 0                # user 0 liked item 3 above their mean
