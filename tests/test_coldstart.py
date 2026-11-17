import numpy as np
import pandas as pd

from src.coldstart import Fallback, truncate_histories


def make_train():
    rng = np.random.default_rng(0)
    counts = [1, 4, 10, 30]
    rows = [(u, i, 3.0 + (i % 3)) for u, c in enumerate(counts) for i in rng.choice(50, c, replace=False)]
    return pd.DataFrame(rows, columns=["u", "i", "rating"]).astype({"u": "int32", "i": "int32"})


def test_truncate_keeps_at_most_n_per_user_and_is_nested():
    tr = make_train()
    c3 = truncate_histories(tr, 4, 50, 3)
    c10 = truncate_histories(tr, 4, 50, 10)
    assert np.diff(c3.indptr).tolist() == [1, 3, 3, 3]            # users with fewer than N keep everything
    assert np.diff(c10.indptr).tolist() == [1, 4, 10, 10]
    assert (c10[c3.nonzero()] != 0).all()                         # the N=3 ratings are a subset of the N=10 ones
    full = truncate_histories(tr, 4, 50, 1000)
    assert full.nnz == len(tr)
    r = full[2, tr[tr.u == 2].i.iloc[0]]
    assert r == tr[(tr.u == 2)].rating.iloc[0]                    # ratings keep their values


def test_fallback_replaces_uninformative_rows_with_popularity():
    pop = np.array([5.0, 1.0, 3.0], dtype=np.float32)
    scores = np.array([[0.0, 0.0, 0.0], [0.2, 0.9, 0.1]], dtype=np.float32)
    fb = Fallback(lambda users: scores[users], pop)
    out = fb(np.array([0, 1]))
    assert out[0].tolist() == pop.tolist() and out[1].tolist() == scores[1].tolist()
    assert fb.frac == 0.5
