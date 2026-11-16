"""Parse the Netflix Prize combined_data_*.txt files into Parquet.

Input format (per file): a movie header line "<movieId>:" followed by
"<customerId>,<rating>,<YYYY-MM-DD>" rows for that movie.

Output: data/processed/netflix_prize/ratings_part{1..4}.parquet with columns
movieId (int16), userId (int32), rating (int8), date (datetime64).
Also prints counts, rating distribution, date range and null/duplicate checks.

Run from the project root:  python src/parse_netflix_prize.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "netflix_prize"
OUT = ROOT / "data" / "processed" / "netflix_prize"


def parse_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, names=["a", "rating", "date"],
        dtype={"a": "string", "rating": "float32", "date": "string"},
    )
    is_header = df["rating"].isna()
    # Header rows look like "123:" in column a; ratings rows have a numeric id.
    movie = df["a"].where(is_header).str.rstrip(":").astype("float64")
    df["movieId"] = movie.ffill()
    df = df.loc[~is_header].copy()
    out = pd.DataFrame({
        "movieId": df["movieId"].astype("int16"),
        "userId": df["a"].astype("int32"),
        "rating": df["rating"].astype("int8"),
        "date": pd.to_datetime(df["date"], format="%Y-%m-%d"),
    })
    return out.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "nulls": 0, "bad_rating": 0}
    users, movies, rating_counts = set(), set(), np.zeros(6, dtype=np.int64)
    dmin, dmax = None, None
    for i in range(1, 5):
        src = RAW / f"combined_data_{i}.txt"
        print(f"parsing {src.name} ...", flush=True)
        part = parse_file(src)
        part.to_parquet(OUT / f"ratings_part{i}.parquet", index=False)
        stats["rows"] += len(part)
        stats["nulls"] += int(part.isna().sum().sum())
        stats["bad_rating"] += int((~part["rating"].between(1, 5)).sum())
        rating_counts += np.bincount(part["rating"].to_numpy(), minlength=6)
        users.update(part["userId"].unique().tolist())
        movies.update(part["movieId"].unique().tolist())
        lo, hi = part["date"].min(), part["date"].max()
        dmin = lo if dmin is None else min(dmin, lo)
        dmax = hi if dmax is None else max(dmax, hi)
        print(f"  {len(part):,} rows", flush=True)
        del part

    print("\n=== Netflix Prize summary ===")
    print(f"ratings:        {stats['rows']:,}")
    print(f"users:          {len(users):,}")
    print(f"movies:         {len(movies):,}")
    print(f"null values:    {stats['nulls']}")
    print(f"ratings not in 1..5: {stats['bad_rating']}")
    print(f"date range:     {dmin.date()} -> {dmax.date()}")
    print(f"density:        {stats['rows'] / (len(users) * len(movies)):.4%}")
    for r in range(1, 6):
        print(f"  rating {r}: {rating_counts[r]:,}")
    print("(duplicate user-movie pairs are checked in the next step; they need a global pass.)")


if __name__ == "__main__":
    main()
