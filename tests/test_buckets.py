import pandas as pd

from src.buckets import relative_to_popularity


def test_relative_to_popularity_divides_by_popularity_in_same_bucket():
    rows = []
    for b, pop, m in (("<30", 0.1, 0.3), ("30-59", 0.2, 0.2), ("60-119", 0.4, 0.2), ("120-299", 0.5, 0.5), ("300+", 1.0, 0.5)):
        rows.append(dict(dataset="d", method="Popularity (most rated)", bucket=b, metric="ndcg@10", value=pop))
        rows.append(dict(dataset="d", method="M", bucket=b, metric="ndcg@10", value=m))
    md = relative_to_popularity(pd.DataFrame(rows))
    assert "| M | 3.00 | 1.00 | 0.50 | 1.00 | 0.50 |" in md
    assert "| Popularity" not in md
