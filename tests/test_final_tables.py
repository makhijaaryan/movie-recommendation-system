import json

import pandas as pd
import pytest

from src.final_tables import build, select_row, to_markdown


def make_results():
    def row(model, params, **m):
        base = {"dataset": "d", "split": "test", "model": model, "params": json.dumps(params), "n_users": 10, "seed": 42}
        for k in (10,):
            for name in ("precision", "recall", "hit", "ndcg"):
                base[f"{name}@{k}"] = m.get(f"{name}@{k}", float("nan"))
        base["rmse"] = m.get("rmse", float("nan"))
        base["mae"] = m.get("mae", float("nan"))
        return base
    return pd.DataFrame([
        row("a", {"k": 1}, rmse=0.9, mae=0.7, **{"ndcg@10": 0.1}),
        row("a", {"k": 2}, rmse=0.8, mae=0.6, **{"ndcg@10": 0.3}),
        row("b", {"k": 1.0}, **{"ndcg@10": 0.2}),
        row("dup", {"k": 1}), row("dup", {"k": 1}),
    ])


def test_select_row_matches_exactly_one_and_compares_numbers():
    r = make_results()
    assert select_row(r, "d", {"model": "a", "params": {"k": 2}})["rmse"] == 0.8
    assert select_row(r, "d", {"model": "b", "params": {"k": 1}})["ndcg@10"] == 0.2   # 1 == 1.0


def test_select_row_errors_on_no_match_or_ambiguity():
    r = make_results()
    with pytest.raises(ValueError):
        select_row(r, "d", {"model": "a", "params": {"k": 99}})
    with pytest.raises(ValueError):
        select_row(r, "d", {"model": "dup", "params": {"k": 1}})
    with pytest.raises(ValueError):
        select_row(r, "d", {"model": "a"})              # both k=1 and k=2 match


def test_build_takes_rating_and_ranking_from_different_rows_and_bolds_best():
    cfg = {"ks": [10], "datasets": {"d": {
        "A": {"rating": {"model": "a", "params": {"k": 2}}, "ranking": {"model": "a", "params": {"k": 1}}},
        "B": {"ranking": {"model": "b"}},
    }}}
    df = build(make_results(), cfg)
    a = df[df.method == "A"].iloc[0]
    assert a["rmse"] == 0.8 and a["ndcg@10"] == 0.1          # RMSE from k=2, ranking from k=1
    md = to_markdown(df, [10])
    assert "**0.8000**" in md and "**0.2000**" in md          # best RMSE (A) and best NDCG (B)
    assert "| B | – | –" in md                                # B has no rating metrics
