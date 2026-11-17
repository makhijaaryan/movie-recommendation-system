import numpy as np
import pandas as pd

from src.data import clean, index_ids, k_core, sample_users, split_per_user, to_csr


def toy(n_users=50, n_items=30, per_user=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        for i in items:
            rows.append((u + 100, i + 7, float(rng.integers(1, 6)), int(rng.integers(0, 10**6))))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "ts"])


def test_clean_removes_bad_rows_and_duplicates():
    df = pd.DataFrame({"userId": [1, 1, 2, 3], "movieId": [5, 5, 6, 7],
                       "rating": [3.0, 4.0, 9.0, np.nan], "ts": [1, 2, 3, 4]})
    out, rep = clean(df, 1.0, 5.0)
    assert rep["nulls"] == 1 and rep["out_of_range"] == 1 and rep["duplicates"] == 1
    assert len(out) == 1 and out.iloc[0]["rating"] == 4.0   # keeps the latest duplicate


def test_k_core_is_stable_and_respects_thresholds():
    df = k_core(toy(), 10, 15)
    assert (df.groupby("userId").size() >= 10).all()
    assert (df.groupby("movieId").size() >= 15).all()


def test_sample_users_is_deterministic():
    df = toy()
    a, b = sample_users(df, 20, 42), sample_users(df, 20, 42)
    assert a["userId"].nunique() == 20 and a.equals(b)


def test_index_ids_contiguous():
    df, uids, iids = index_ids(toy())
    assert df["u"].max() == len(uids) - 1 and df["i"].max() == len(iids) - 1
    assert set(df["u"]) == set(range(len(uids)))


def test_split_partitions_and_keeps_train_per_user():
    df, _, _ = index_ids(toy(per_user=20))
    lab = split_per_user(df, seed=42)
    assert set(lab.unique()) == {"train", "val", "test"}
    per_user = pd.crosstab(df["u"], lab)
    assert (per_user["train"] >= 1).all() and (per_user["val"] >= 1).all() and (per_user["test"] >= 1).all()
    # 20 ratings -> 2 test, 2 val, 16 train
    assert (per_user["test"] == 2).all() and (per_user["val"] == 2).all()


def test_split_is_reproducible_and_seed_dependent():
    df, _, _ = index_ids(toy())
    assert split_per_user(df, 42).equals(split_per_user(df, 42))
    assert not split_per_user(df, 42).equals(split_per_user(df, 7))


def test_to_csr_shape_and_values():
    df, uids, iids = index_ids(toy())
    m = to_csr(df, len(uids), len(iids))
    assert m.shape == (len(uids), len(iids)) and m.nnz == len(df)
