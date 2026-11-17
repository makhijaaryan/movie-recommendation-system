import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.eval import (add_history_bucket, evaluate_rating, evaluate_topk, log_result, mae, rmse,
                      topk_metrics_from_lists)


def test_rmse_mae_known_values():
    y, p = [3, 4, 5], [3, 5, 3]
    assert rmse(y, p) == pytest.approx(np.sqrt((0 + 1 + 4) / 3))
    assert mae(y, p) == pytest.approx(1.0)


def test_evaluate_rating_clips():
    df = pd.DataFrame({"u": [0, 1], "i": [0, 1], "rating": [5.0, 1.0]})
    out = evaluate_rating(lambda u, i: np.array([99.0, -9.0]), df, rating_range=(1.0, 5.0))
    assert out["rmse"] == 0.0


def test_topk_metrics_hand_computed():
    # 6 items; ranked list [2, 0, 4, 1, 3, 5]; relevant = {2, 4}
    topk = np.array([[2, 0, 4, 1, 3, 5]])
    rel = np.zeros((1, 6), bool)
    rel[0, [2, 4]] = True
    m = topk_metrics_from_lists(topk, rel, ks=(1, 3))
    assert m["precision@1"][0] == 1.0 and m["recall@1"][0] == 0.5
    assert m["precision@3"][0] == pytest.approx(2 / 3) and m["recall@3"][0] == 1.0
    dcg = 1 / np.log2(2) + 1 / np.log2(4)           # hits at ranks 1 and 3
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg@3"][0] == pytest.approx(dcg / idcg)
    assert m["hit@1"][0] == 1.0


def test_hit_zero_when_no_hits():
    m = topk_metrics_from_lists(np.array([[0, 1]]), np.array([[False, False, True]]), ks=(2,))
    assert m["hit@2"][0] == 0 and m["ndcg@2"][0] == 0


def make_toy():
    # 3 users, 5 items. train: u0 saw {0}, u1 saw {1}, u2 saw {2}
    train = sp.csr_matrix(([1.0, 1.0, 1.0], ([0, 1, 2], [0, 1, 2])), shape=(3, 5))
    test = pd.DataFrame({"u": [0, 1, 2, 2], "i": [3, 4, 3, 4], "rating": [5.0, 4.0, 4.5, 2.0]})
    return train, test


def test_evaluate_topk_exclusion_of_seen_items():
    train, test = make_toy()
    scores = np.zeros((3, 5))
    scores[0, 3] = scores[1, 4] = scores[2, 3] = 10     # the relevant items
    scores[:, 0] = 100                                   # item 0 is top-scored for everyone
    res = evaluate_topk(lambda b: scores[b], test, train, 5, ks=(1, 5))
    assert res["n_users"] == 3
    # u0 saw item 0 -> excluded, so its top-1 is the relevant item 3 (hit).
    # u1, u2 have item 0 as an unseen top-1 (not relevant) -> miss.
    assert res["hit@1"] == pytest.approx(1 / 3)
    assert res["hit@5"] == 1.0


def test_users_without_relevant_items_are_skipped():
    train, test = make_toy()
    test.loc[test.u == 1, "rating"] = 1.0                # u1 has no relevant item
    res = evaluate_topk(lambda b: np.zeros((len(b), 5)), test, train, 5, ks=(5,))
    assert res["n_users"] == 2


def test_history_bucket():
    dense = np.ones((2, 100))
    dense[1, 40:] = 0
    train = sp.csr_matrix(dense)
    per_user = pd.DataFrame({"u": [0, 1], "ndcg@10": [0.1, 0.2]})
    out = add_history_bucket(per_user, train)
    assert list(out["bucket"].astype(str)) == ["60-119", "30-59"]


def test_log_result_appends(tmp_path):
    p = tmp_path / "r.csv"
    log_result("toy", "m", "val", {"k": 1}, {"rmse": 1.5}, path=p)
    log_result("toy", "m2", "val", {"k": 2}, {"rmse": 1.2, "mae": 0.9}, path=p)   # new column
    df = pd.read_csv(p)
    assert len(df) == 2 and "mae" in df.columns
