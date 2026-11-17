"""Data pipeline: load -> clean -> filter -> sample -> index -> split -> save.

Datasets (names used everywhere in the project):
    ml_small  MovieLens latest-small (development only)
    ml_25m    MovieLens 25M          (main comparison)
    netflix   Netflix Prize subsample (second dataset)

Outputs go to data/processed/<name>/:
    train.parquet, val.parquet, test.parquet   columns: u, i, rating, ts
    train_csr.npz                              scipy CSR user-item matrix (train)
    user_ids.npy, item_ids.npy                 index -> original id
    meta.json                                  parameters + statistics

Run from the project root:
    python -m src.data --dataset ml_small
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEED = 42

# Per-dataset pipeline parameters (see DIFFERENCE.md D8, D9, D10).
CONFIG = {
    "ml_small": dict(min_user=20, min_item=5, sample_users=None),
    "ml_25m": dict(min_user=20, min_item=50, sample_users=None),
    "netflix": dict(min_user=20, min_item=50, sample_users=50_000),
}


# --------------------------------------------------------------------------- load
def load_raw(name: str) -> pd.DataFrame:
    """Return raw ratings with columns userId, movieId, rating, ts (int64 seconds)."""
    if name in ("ml_small", "ml_25m"):
        folder = "ml-latest-small" if name == "ml_small" else "ml-25m"
        df = pd.read_csv(
            RAW / "movielens" / folder / "ratings.csv",
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        ).rename(columns={"timestamp": "ts"})
        return df
    if name == "netflix":
        files = sorted(glob.glob(str(PROC / "netflix_prize" / "ratings_part*.parquet")))
        if not files:
            raise FileNotFoundError("Run `python src/parse_netflix_prize.py` first.")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["ts"] = df["date"].astype("datetime64[s]").astype("int64")
        df["movieId"] = df["movieId"].astype("int32")
        df["rating"] = df["rating"].astype("float32")
        return df[["userId", "movieId", "rating", "ts"]]
    raise ValueError(f"unknown dataset {name!r}")


# -------------------------------------------------------------------------- clean
def clean(df: pd.DataFrame, lo: float, hi: float) -> tuple[pd.DataFrame, dict]:
    """Drop nulls, out-of-range ratings, duplicate (user, item) pairs (keep latest)."""
    n0 = len(df)
    df = df.dropna()
    n_null = n0 - len(df)
    in_range = df["rating"].between(lo, hi)
    n_bad = int((~in_range).sum())
    df = df[in_range]
    n1 = len(df)
    df = df.sort_values("ts").drop_duplicates(["userId", "movieId"], keep="last")
    report = dict(rows_in=n0, nulls=n_null, out_of_range=n_bad, duplicates=n1 - len(df), rows_out=len(df))
    return df.reset_index(drop=True), report


# ------------------------------------------------------------------------- filter
def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively drop users with < min_user and items with < min_item ratings until stable."""
    while True:
        n = len(df)
        uc = df["userId"].map(df["userId"].value_counts())
        df = df[uc >= min_user]
        ic = df["movieId"].map(df["movieId"].value_counts())
        df = df[ic >= min_item]
        if len(df) == n:
            return df.reset_index(drop=True)


def sample_users(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    users = np.sort(df["userId"].unique())
    if n >= len(users):
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=n, replace=False)
    return df[df["userId"].isin(keep)].reset_index(drop=True)


# -------------------------------------------------------------------------- index
def index_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Map userId/movieId to contiguous 0-based u/i. Returns (df, user_ids, item_ids)."""
    u, user_ids = pd.factorize(df["userId"], sort=True)
    i, item_ids = pd.factorize(df["movieId"], sort=True)
    out = pd.DataFrame({
        "u": u.astype("int32"),
        "i": i.astype("int32"),
        "rating": df["rating"].to_numpy(dtype="float32"),
        "ts": df["ts"].to_numpy(dtype="int64"),
    })
    return out, np.asarray(user_ids), np.asarray(item_ids)


# --------------------------------------------------------------------------- split
def split_per_user(df: pd.DataFrame, seed: int = SEED, frac_val: float = 0.1,
                   frac_test: float = 0.1) -> pd.Series:
    """Per-user random split. Returns a Series of 'train'/'val'/'test' aligned to df.

    For a user with n ratings: n_test = max(1, int(frac_test*n)), n_val likewise;
    the rest is train, so every user with n >= 3 keeps training data.
    """
    rng = np.random.default_rng(seed)
    key = rng.random(len(df))
    order = np.lexsort((key, df["u"].to_numpy()))      # sort by user, then random key
    u_sorted = df["u"].to_numpy()[order]
    n_users = int(u_sorted.max()) + 1
    counts = np.bincount(u_sorted, minlength=n_users)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = np.arange(len(df)) - starts[u_sorted]         # rank within the user's shuffled list
    n = counts[u_sorted]
    n_test = np.maximum(1, (frac_test * n).astype(int))
    n_val = np.maximum(1, (frac_val * n).astype(int))
    label_sorted = np.where(pos < n_test, "test", np.where(pos < n_test + n_val, "val", "train"))
    labels = np.empty(len(df), dtype=object)
    labels[order] = label_sorted
    return pd.Series(labels, index=df.index)


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    return sp.csr_matrix((df["rating"].to_numpy(dtype="float32"), (df["u"], df["i"])),
                         shape=(n_users, n_items))


# ----------------------------------------------------------------------- pipeline
def build(name: str, seed: int = SEED) -> dict:
    cfg = CONFIG[name]
    lo, hi = (0.5, 5.0) if name.startswith("ml") else (1.0, 5.0)
    out_dir = PROC / name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(name)
    df, clean_report = clean(raw, lo, hi)
    del raw
    stats = {"clean": clean_report}

    if name == "netflix":
        # D8: user >=20 first, then sample users, then enforce the k-core on the sample.
        df = k_core(df, cfg["min_user"], 1)
        stats["after_user_filter"] = dict(users=int(df.userId.nunique()), ratings=len(df))
        df = sample_users(df, cfg["sample_users"], seed)
        stats["after_sample"] = dict(users=int(df.userId.nunique()), ratings=len(df))
    df = k_core(df, cfg["min_user"], cfg["min_item"])

    df, user_ids, item_ids = index_ids(df)
    n_users, n_items = len(user_ids), len(item_ids)
    df["split"] = split_per_user(df, seed)

    parts = {s: df[df.split == s].drop(columns="split").reset_index(drop=True)
             for s in ("train", "val", "test")}
    for s, p in parts.items():
        p.to_parquet(out_dir / f"{s}.parquet", index=False)
    sp.save_npz(out_dir / "train_csr.npz", to_csr(parts["train"], n_users, n_items))
    np.save(out_dir / "user_ids.npy", user_ids)
    np.save(out_dir / "item_ids.npy", item_ids)

    total = len(df)
    train_items = set(parts["train"]["i"].unique().tolist())
    stats.update(
        dataset=name, seed=seed, config=cfg,
        users=n_users, items=n_items, ratings=total,
        density=total / (n_users * n_items),
        split_sizes={s: len(p) for s, p in parts.items()},
        val_test_items_unseen_in_train={
            s: int((~parts[s]["i"].isin(train_items)).sum()) for s in ("val", "test")},
        rating_mean_train=float(parts["train"]["rating"].mean()),
    )
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2, default=int))
    return stats


# ----------------------------------------------------------------------- loaders
def load_split(name: str, which: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / name / f"{which}.parquet")


def load_train_csr(name: str) -> sp.csr_matrix:
    return sp.load_npz(PROC / name / "train_csr.npz")


def load_id_maps(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(PROC / name / "user_ids.npy"), np.load(PROC / name / "item_ids.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIG), required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    print(json.dumps(build(args.dataset, args.seed), indent=2, default=int))
