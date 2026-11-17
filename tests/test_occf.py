import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.eval import evaluate_topk
from src.occf import OCCF, knn_graph, positives_csr


def planted(n_users=200, n_items=60, groups=4, p_in=0.5, p_out=0.02, seed=0):
    """Users and items in `groups` communities; users mostly like items of their own community."""
    rng = np.random.default_rng(seed)
    ug = rng.integers(0, groups, n_users)
    ig = rng.integers(0, groups, n_items)
    prob = np.where(ug[:, None] == ig[None, :], p_in, p_out)
    likes = rng.random((n_users, n_items)) < prob
    return likes, ug, ig


def split(likes, seed=1):
    rng = np.random.default_rng(seed)
    u, i = np.nonzero(likes)
    hold = rng.random(len(u)) < 0.25
    tr = pd.DataFrame({"u": u[~hold], "i": i[~hold], "rating": 5.0})
    te = pd.DataFrame({"u": u[hold], "i": i[hold], "rating": 5.0})
    return tr, te


def test_positives_threshold():
    df = pd.DataFrame({"u": [0, 0, 1], "i": [0, 1, 1], "rating": [5.0, 3.0, 4.0]})
    m = positives_csr(df, 2, 2)
    assert m.nnz == 2 and m[0, 1] == 0 and m[1, 1] == 1


def test_wrmf_recovers_planted_structure_and_beats_popularity():
    likes, ug, ig = planted()
    tr, te = split(likes)
    nu, ni = likes.shape
    pos = positives_csr(tr, nu, ni)
    m = OCCF(k=8, alpha=10, lam=1.0, iters=10).fit(pos, verbose=False)
    seen = sp.csr_matrix((np.ones(len(tr)), (tr.u, tr.i)), shape=(nu, ni))
    res = evaluate_topk(m.score_fn, te, seen, ni, ks=(5,))
    pop = np.asarray(pos.sum(0)).ravel()
    base = evaluate_topk(lambda b: np.tile(pop, (len(b), 1)), te, seen, ni, ks=(5,))
    assert res["ndcg@5"] > 1.3 * base["ndcg@5"]


def test_fit_is_deterministic():
    likes, *_ = planted(n_users=60, n_items=30)
    tr, _ = split(likes)
    pos = positives_csr(tr, *likes.shape)
    a = OCCF(k=4, iters=3).fit(pos, seed=3, verbose=False)
    b = OCCF(k=4, iters=3).fit(pos, seed=3, verbose=False)
    assert np.allclose(a.P, b.P) and np.allclose(a.Q, b.Q)


def test_knn_graph_finds_same_community_and_excludes_self():
    likes, ug, ig = planted(p_in=0.7, p_out=0.01)
    m = sp.csr_matrix(likes.astype(np.float32))
    idx, w = knn_graph(m, top_m=5, threads=2, block=64)
    assert not np.any(idx == np.arange(len(idx))[:, None])
    frac_same = np.mean(ug[idx] == ug[:, None])
    assert frac_same > 0.9 and np.all(w >= 0)


def test_graph_regularisation_runs_and_pulls_neighbours_together():
    likes, ug, ig = planted()
    tr, _ = split(likes)
    nu, ni = likes.shape
    pos = positives_csr(tr, nu, ni)
    ugr = knn_graph(pos, 5, threads=2, block=64)
    igr = knn_graph(pos.T.tocsr(), 5, threads=2, block=64)
    plain = OCCF(k=8, alpha=10, lam=1.0, iters=8).fit(pos, verbose=False)
    dual = OCCF(k=8, alpha=10, lam=1.0, lam_u=20.0, lam_i=20.0, iters=8).fit(pos, ugr, igr, verbose=False)

    def graph_dist(P, graph):
        idx, w = graph
        return float(np.mean(np.sum((P[:, None, :] - P[idx]) ** 2, axis=2) * (w > 0))) / float(np.mean(np.sum(P ** 2, axis=1)))
    assert np.all(np.isfinite(dual.P)) and graph_dist(dual.P, ugr) < graph_dist(plain.P, ugr)
