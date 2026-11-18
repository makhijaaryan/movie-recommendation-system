"""Qualitative success / failure examples (task 6.6), MovieLens 25M.

Models (frozen ranking configurations): popularity, item CF, user CF, MF (biased), one-class (dual).
A fixed random sample of eligible users (seed 42) is scored; users are then picked BY RULE (never by hand),
one per case, with a seeded random draw among all users that satisfy the rule. The report states how many
users satisfy each rule, so the examples can be judged against how common each situation is.

For each example: the user's highest-rated training movies, the held-out movies they liked (rating >= 4), and each
model's top-10 list with  ✓ = rated >= 4 in the held-out test split,  ✗ = rated < 4 there,  · = no held-out rating
(unknown, NOT a dislike). Only the held-out ratings are known, so a '·' recommendation may still be a good one.

Output: reports/examples.md
Run:    python -m src.examples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.config import SEED
from src.diagnostics import load_genres, top_lists
from src.final_run import fit_key, fit_model, job_key, set_k

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
REPORT = ROOT / "reports" / "examples.md"
MODELS = ["Popularity (most rated)", "Item-based CF", "User-based CF", "MF (biased)", "One-class dual-regularized"]
SHORT = {"Popularity (most rated)": "Popularity", "Item-based CF": "Item CF", "User-based CF": "User CF",
         "MF (biased)": "MF (biased)", "One-class dual-regularized": "One-class"}
POP, ITEM, USER, MF, OC = MODELS
K = 10


def load_titles(dataset: str = "ml_25m") -> list[str]:
    movies = pd.read_csv(ROOT / "data" / "raw" / "movielens" / "ml-25m" / "movies.csv")
    t = dict(zip(movies.movieId, movies.title))
    _, item_ids = data.load_id_maps(dataset)
    return [t.get(int(m), f"movie {m}") for m in item_ids]


def pick_cases(feat: pd.DataFrame, hits: dict, n_rel: np.ndarray, seed: int = SEED):
    """feat: per-user n_train, niche score (mean popularity percentile of the user's train items). hits: label -> hits@10.
    Returns {case: (user_row_index or None, n_candidates)}."""
    nt, ns = feat["n_train"].to_numpy(), feat["niche"].to_numpy()
    hi = np.quantile(ns, 0.9)
    rules = {
        "S1 success, heavy user with mainstream taste": (nt >= 300) & (ns <= np.median(ns[nt >= 300])) & (hits[OC] >= 5),
        "S2 success, light user (<30 ratings), popularity misses": (nt < 30) & (hits[OC] >= 2) & (hits[POP] == 0),
        "S3 success, niche-taste user": (ns >= hi) & (hits[OC] >= 3) & (hits[POP] <= 1),
        "F1 failure, every model gets zero hits": (nt >= 60) & (n_rel >= 3) & np.all([hits[m] == 0 for m in MODELS], 0),
        "F2 failure, popularity gets more hits than one-class": (hits[POP] >= 2) & (hits[POP] > hits[OC]),
        "F2b strict version: popularity >= 3 hits, one-class none": (hits[POP] >= 3) & (hits[OC] == 0),
        "F3 failure, MF misses where the other personalised models succeed":
            (hits[MF] == 0) & (hits[OC] >= 4) & (hits[USER] >= 3) & (hits[ITEM] >= 2),
    }
    rng = np.random.default_rng(seed)
    out = {}
    for case, mask in rules.items():
        idx = np.nonzero(mask)[0]
        out[case] = (int(rng.choice(idx)) if len(idx) else None, int(len(idx)))
    return out


def main(n_users: int, config_path: str) -> None:
    dataset = "ml_25m"
    methods = json.loads(Path(config_path).read_text())["datasets"][dataset]
    train, val, test_full = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    nu, ni = tr.shape
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    eligible = np.sort(test_full.loc[test_full["rating"] >= 4.0, "u"].unique())
    users = np.sort(np.random.default_rng(SEED).choice(eligible, n_users, replace=False))
    titles, genres = load_titles(dataset), load_genres(dataset)
    user_ids, _ = data.load_id_maps(dataset)
    ctx = dict(dataset=dataset, train=train, val=val, n_users=nu, n_items=ni, tr=tr)

    jobs: dict[str, dict] = {}
    for label in MODELS:
        spec = methods[label]["ranking"]
        jobs.setdefault(job_key(spec), dict(spec=spec, labels=[])).setdefault("labels", []).append(label)
    lists, model, mkey = {}, None, None
    for j in sorted(jobs.values(), key=lambda j: fit_key(j["spec"])):
        if fit_key(j["spec"]) != mkey:
            model = None
            model = fit_model(j["spec"], ctx)
            mkey = fit_key(j["spec"])
        set_k(model, j["spec"], "ranking")
        for label in j["labels"]:
            lists[label] = top_lists(model, users, trval, K)
        print("lists ready:", j["labels"], flush=True)

    # held-out information
    liked = test_full[test_full["rating"] >= 4.0].groupby("u")["i"].apply(set).to_dict()
    held = {u: dict(zip(g["i"], g["rating"])) for u, g in test_full[test_full["u"].isin(users)].groupby("u")}
    hits = {m: np.array([len(set(L[n]) & liked.get(u, set())) for n, u in enumerate(users)]) for m, L in lists.items()}
    n_rel = np.array([len(liked.get(u, ())) for u in users])

    count = np.bincount(train["i"], minlength=ni)
    pop_pct = np.empty(ni)
    pop_pct[np.argsort(-count, kind="stable")] = np.arange(ni) / ni
    feat = pd.DataFrame({"n_train": np.diff(tr.indptr)[users],
                         "niche": [pop_pct[tr.indices[tr.indptr[u]:tr.indptr[u + 1]]].mean() for u in users]})
    cases = pick_cases(feat, hits, n_rel)

    tr_sorted = train.sort_values(["u", "rating", "ts"], ascending=[True, False, False])
    out = [f"# Qualitative examples (MovieLens 25M, seed 42)\n",
           f"Sample: {len(users):,} random eligible users. Models: " + ", ".join(SHORT[m] for m in MODELS) + ". "
           "Cases are picked by the stated rules with a seeded random draw among all users meeting the rule (the "
           "count is shown), not by hand. ✓ = rated ≥ 4 in the held-out test split, ✗ = rated < 4 there, "
           "· = no held-out rating (unknown, not a dislike; the recommendation may still be good).\n",
           "## How common is each case?\n", "| Case | Users meeting the rule | Share of sample |", "|:--|---:|---:|"]
    out += [f"| {c} | {n} | {n / len(users):.1%} |" for c, (_, n) in cases.items()]
    out.append("")
    zero = {SHORT[m]: float(np.mean(hits[m] == 0)) for m in MODELS}
    out.append("Share of sampled users with **no hit in the top 10**: " + ", ".join(f"{k} {v:.0%}" for k, v in zero.items())
               + ".\n")

    for case, (n, cnt) in cases.items():
        out.append(f"## {case}\n")
        if n is None:
            out.append("_No user in the sample meets this rule._\n")
            continue
        u = int(users[n])
        mine = tr_sorted[tr_sorted["u"] == u]
        fav = mine.head(8)
        liked_train_genres = pd.Series([g for i in mine[mine.rating >= 4]["i"] for g in genres[i]]).value_counts().head(4)
        held_liked = [i for i, r in held.get(u, {}).items() if r >= 4]
        held_genres = pd.Series([g for i in held_liked for g in genres[i]]).value_counts().head(4)
        out.append(f"**User {int(user_ids[u])}** — {int(feat.n_train[n])} training ratings, "
                   f"{len(held_liked)} held-out liked movies, niche score {feat.niche[n]:.2f} "
                   f"(0 = watches only the most-rated movies, 1 = only the least-rated).\n")
        out.append("*Favourite genres in history:* " + ", ".join(f"{g} ({c})" for g, c in liked_train_genres.items())
                   + ".  *Genres of held-out liked movies:* " + ", ".join(f"{g} ({c})" for g, c in held_genres.items())
                   + ".\n")
        out.append("*Highest-rated training movies:* " + "; ".join(
            f"{titles[r.i]} ({r.rating:g})" for r in fav.itertuples()) + ".\n")
        out.append("*Held-out liked movies:* " + "; ".join(titles[i] for i in held_liked[:12])
                   + (" …" if len(held_liked) > 12 else "") + ".\n")
        out.append("| # | " + " | ".join(f"{SHORT[m]} ({hits[m][n]} hits)" for m in MODELS) + " |")
        out.append("|--:|" + ":--|" * len(MODELS))
        for r in range(K):
            cells = []
            for m in MODELS:
                i = int(lists[m][n][r])
                rr = held.get(u, {}).get(i)
                mark = "·" if rr is None else ("✓" if rr >= 4 else "✗")
                cells.append(f"{mark} {titles[i]} <sub>[{count[i]:,}]</sub>")
            out.append(f"| {r + 1} | " + " | ".join(cells) + " |")
        out.append("\n<sub>[n] = number of training ratings of the movie (popularity).</sub>\n")
    REPORT.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args()
    main(a.users, a.config)
