import numpy as np

from src.diagnostics import gini, list_metrics


def test_gini_extremes():
    assert abs(gini(np.array([5, 5, 5, 5]))) < 1e-9
    assert gini(np.array([0, 0, 0, 100])) > 0.7


def test_list_metrics_on_a_toy_catalogue():
    count = np.array([100, 50, 10, 1, 1])               # item 0 most rated, items 3,4 least
    mean = np.array([4.0, 3.0, 5.0, 2.0, 1.0])
    pop_only = np.array([[0, 1], [0, 1], [0, 1]])        # everybody gets the two most popular items
    m = list_metrics(pop_only, count, mean)
    assert m["coverage"] == 2 / 5 and m["tail_share"] == 0 and m["head_share"] == 0.5
    assert m["item_mean"] == 3.5 and m["median_ratings"] == 75
    diverse = np.array([[0, 3], [1, 4], [2, 3]])
    d = list_metrics(diverse, count, mean, genres=[{"a"}, {"b"}, {"a", "c"}, {"d"}, {"e"}])
    assert d["coverage"] == 1.0 and d["tail_share"] > m["tail_share"] and d["novelty"] > m["novelty"]
    assert d["genres"] == (2 + 2 + 3) / 3
