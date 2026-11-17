"""Exploratory data analysis (proposal Week 3): popular movies, active users,
sparsity and rating distributions for MovieLens 25M and the Netflix Prize subsample,
plus a look at the Kaggle Netflix titles metadata.

Writes figures to reports/figures/ and numbers to reports/eda_summary.json and
reports/eda_tables.md.  Run from the project root:  python -m src.eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data
from src.eval import BUCKET_EDGES, BUCKET_LABELS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RAW = ROOT / "data" / "raw"
DATASETS = ("ml_25m", "netflix")
LABEL = {"ml_25m": "MovieLens 25M", "netflix": "Netflix Prize (50k-user sample)"}
COLOR = {"ml_25m": "#1f77b4", "netflix": "#e6550d"}   # blue / orange, colour-blind safe pair
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def load_all(name: str) -> pd.DataFrame:
    return pd.concat([data.load_split(name, s) for s in ("train", "val", "test")], ignore_index=True)


def titles(name: str, item_ids: np.ndarray) -> pd.Series:
    """Series index = item index (0-based), value = title."""
    if name == "ml_25m":
        m = pd.read_csv(RAW / "movielens" / "ml-25m" / "movies.csv").set_index("movieId")["title"]
    else:
        rows = []
        with open(RAW / "netflix_prize" / "movie_titles.csv", encoding="latin-1") as f:
            for line in f:
                mid, year, t = line.rstrip("\n").split(",", 2)
                rows.append((int(mid), f"{t} ({year})"))
        m = pd.Series(dict(rows))
    return pd.Series(m.reindex(item_ids).to_numpy(), index=np.arange(len(item_ids)))


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, md = {}, []
    frames = {n: load_all(n) for n in DATASETS}
    maps = {n: data.load_id_maps(n) for n in DATASETS}

    # ---------------------------------------------------------------- summary table
    rows = []
    for n, df in frames.items():
        nu, ni = df.u.nunique(), df.i.nunique()
        per_u = df.groupby("u").size()
        per_i = df.groupby("i").size().sort_values(ascending=False)
        top1 = per_i.iloc[: max(1, int(0.01 * ni))].sum() / len(df)
        top10 = per_i.iloc[: max(1, int(0.10 * ni))].sum() / len(df)
        s = dict(users=nu, items=ni, ratings=len(df), density_pct=100 * len(df) / (nu * ni),
                 mean_rating=float(df.rating.mean()), median_rating=float(df.rating.median()),
                 ratings_per_user_median=float(per_u.median()), ratings_per_user_mean=float(per_u.mean()),
                 ratings_per_user_max=int(per_u.max()), ratings_per_item_median=float(per_i.median()),
                 ratings_per_item_max=int(per_i.max()), share_ratings_top1pct_items=float(top1),
                 share_ratings_top10pct_items=float(top10))
        summary[n] = s
        rows.append(pd.Series(s, name=LABEL[n]))
    tbl = pd.DataFrame(rows).T
    md += ["## Dataset statistics (after filtering, all splits)\n", tbl.to_markdown(floatfmt=",.3f"), ""]

    # ------------------------------------------------------------ figure 1: ratings
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, n in zip(axes, DATASETS):
        vc = frames[n].rating.value_counts(normalize=True).sort_index()
        ax.bar(vc.index.astype(str), 100 * vc.values, color=COLOR[n])
        ax.set(title=LABEL[n], xlabel="Rating", ylabel="% of ratings")
        for x, v in zip(vc.index.astype(str), vc.values):
            ax.text(x, 100 * v, f"{100*v:.0f}", ha="center", va="bottom", fontsize=8)
    savefig(fig, "eda_rating_distribution.png")

    # ----------------------------------------------- figure 2: user activity (long tail)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, n in zip(axes, DATASETS):
        per_u = frames[n].groupby("u").size()
        ax.hist(per_u, bins=np.logspace(np.log10(per_u.min()), np.log10(per_u.max()), 50), color=COLOR[n])
        ax.set(xscale="log", title=LABEL[n], xlabel="Ratings per user (log)", ylabel="Users")
        ax.axvline(per_u.median(), color="k", ls="--", lw=1)
        ax.text(per_u.median() * 1.1, ax.get_ylim()[1] * 0.9, f"median {per_u.median():.0f}", fontsize=8)
    savefig(fig, "eda_user_activity.png")

    # ----------------------------------------------- figure 3: item popularity long tail
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for n in DATASETS:
        per_i = frames[n].groupby("i").size().sort_values(ascending=False).to_numpy()
        ax.plot(np.arange(1, len(per_i) + 1), per_i, color=COLOR[n], label=LABEL[n])
    ax.set(xscale="log", yscale="log", xlabel="Item rank (log)", ylabel="Ratings (log)",
           title="Item popularity: a long tail")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, "eda_item_popularity.png")

    # ------------------------------------------- figure 4: user history buckets (train)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    w = 0.38
    bucket_counts = {}
    for k, n in enumerate(DATASETS):
        tr = data.load_split(n, "train")
        c = tr.groupby("u").size()
        b = pd.cut(c, bins=list(BUCKET_EDGES), labels=list(BUCKET_LABELS), right=False).value_counts().reindex(BUCKET_LABELS)
        bucket_counts[n] = {k_: int(v) for k_, v in b.items()}
        ax.bar(np.arange(len(b)) + (k - 0.5) * w, 100 * b.values / b.sum(), w, color=COLOR[n], label=LABEL[n])
    ax.set_xticks(np.arange(len(BUCKET_LABELS)), BUCKET_LABELS)
    ax.set(xlabel="Training ratings per user (history bucket)", ylabel="% of users",
           title="How much history do users have?")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, "eda_history_buckets.png")
    summary["history_buckets_users"] = bucket_counts

    # ----------------------------------------------------------- figure 5: over time
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, n in zip(axes, DATASETS):
        yr = pd.to_datetime(frames[n].ts, unit="s").dt.year
        g = frames[n].assign(year=yr).groupby("year").rating.agg(["size", "mean"])
        ax.bar(g.index, g["size"] / 1e6, color=COLOR[n], alpha=0.8)
        ax.set(title=LABEL[n], xlabel="Year", ylabel="Ratings (millions)")
        ax2 = ax.twinx()
        ax2.plot(g.index, g["mean"], color="k", marker="o", ms=3, lw=1)
        ax2.set(ylabel="Mean rating", ylim=(3.0, 4.2)); ax2.grid(False)
    savefig(fig, "eda_time.png")

    # ----------------------------------------------------------- top movies tables
    for n in DATASETS:
        df, (uids, iids) = frames[n], maps[n]
        t = titles(n, iids)
        g = df.groupby("i").rating.agg(["size", "mean"])
        top_pop = g.sort_values("size", ascending=False).head(10)
        min_n = 1000 if n == "ml_25m" else 2000
        top_rated = g[g["size"] >= min_n].sort_values("mean", ascending=False).head(10)
        for label, tt in ((f"Most rated — {LABEL[n]}", top_pop),
                          (f"Highest mean rating (≥{min_n} ratings) — {LABEL[n]}", top_rated)):
            out = pd.DataFrame({"title": t.loc[tt.index].to_numpy(), "ratings": tt["size"].to_numpy(),
                                "mean": tt["mean"].round(3).to_numpy()})
            md += [f"## {label}\n", out.to_markdown(index=False), ""]

    # ------------------------------------------------------------ MovieLens genres
    movies = pd.read_csv(RAW / "movielens" / "ml-25m" / "movies.csv")
    uid, iid = maps["ml_25m"]
    gm = movies.set_index("movieId").loc[iid, "genres"].reset_index(drop=True)
    df = frames["ml_25m"]
    genre_of = gm.str.split("|").explode()
    item_stats = df.groupby("i").rating.agg(["size", "sum"])
    ge = pd.DataFrame({"genre": genre_of}).join(item_stats, how="inner")
    gstat = ge.groupby("genre").agg(ratings=("size", "sum"), total=("sum", "sum"))
    gstat["mean_rating"] = gstat.total / gstat.ratings
    gstat = gstat.drop(columns="total").sort_values("ratings", ascending=False)
    gstat = gstat[gstat.index != "(no genres listed)"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(gstat.index[::-1], gstat.ratings[::-1] / 1e6, color=COLOR["ml_25m"])
    ax.set(xlabel="Ratings (millions; a movie counts once per genre)", title="MovieLens 25M: ratings by genre")
    savefig(fig, "eda_ml_genres.png")
    md += ["## MovieLens 25M genres\n", gstat.round(3).to_markdown(floatfmt=",.3f"), ""]

    # ------------------------------------------------- Kaggle Netflix titles metadata
    nt = pd.read_csv(RAW / "netflix_titles" / "netflix_dataset.csv")
    kt = dict(titles=len(nt), movies=int((nt.type == "Movie").sum()), tv_shows=int((nt.type == "TV Show").sum()),
              missing_director_pct=float(100 * nt.director.isna().mean()),
              missing_country_pct=float(100 * nt.country.isna().mean()),
              release_year_min=int(nt.release_year.min()), release_year_max=int(nt.release_year.max()))
    summary["kaggle_netflix_titles"] = kt
    genres = nt.listed_in.str.split(", ").explode().value_counts().head(12)
    countries = nt.country.dropna().str.split(", ").explode().value_counts().head(10)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(nt.release_year, bins=range(1940, 2023, 3), color=COLOR["netflix"])
    axes[0].set(title="Kaggle Netflix titles: release year", xlabel="Year", ylabel="Titles")
    axes[1].barh(genres.index[::-1], genres.values[::-1], color=COLOR["netflix"]); axes[1].set(title="Top genres")
    axes[2].barh(countries.index[::-1], countries.values[::-1], color=COLOR["netflix"]); axes[2].set(title="Top countries")
    savefig(fig, "eda_kaggle_titles.png")
    md += ["## Kaggle Netflix titles (metadata only; no user ratings)\n", pd.Series(kt).to_frame("value").to_markdown(), ""]

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (ROOT / "reports" / "eda_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
