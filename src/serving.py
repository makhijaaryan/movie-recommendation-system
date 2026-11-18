"""Live recommendations for a NEW user on MovieLens 25M (web demo backend).

The user supplies a few ratings; every model scores all movies for that user without retraining:
  popularity      most-rated movies (same list for everybody)
  item CF         item-item similarities from the full training matrix, applied to the user's ratings
  user CF         the user's ratings compared with all training users (neighbour pool = training users)
  MF (biased)     closed-form ridge fold-in (src/foldin.py), item factors fixed
  one-class dual  closed-form ALS user step on the user's ratings >= 4 (src/foldin.py), item factors fixed
The models are the frozen ranking configurations trained on the train split (src/export_models.py).
Users whose ratings give a model no signal (e.g. nothing rated >= 4 for one-class) fall back to popularity and
the response says so (same rule as the cold-start experiment, DIFFERENCE.md D34).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from src import data
from src.cf import ItemKNN, UserKNN
from src.config import processed_dir
from src.foldin import foldin_mf, foldin_occf

ROOT = Path(__file__).resolve().parents[1]
DATASET = "ml_25m"
MODELS = {
    "popularity": "Popularity",
    "item_cf": "Item-based CF",
    "user_cf": "User-based CF",
    "mf": "Matrix factorization",
    "one_class": "One-class CF",
}
ITEM_K, USER_K = 50, 200            # ranking configurations (experiments/final_config.json)


def display_title(raw: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", raw)
    title, year = (m.group(1), int(m.group(2))) if m else (raw, None)
    m2 = re.match(r"^(.*), (The|A|An)$", title)
    if m2:
        title = f"{m2.group(2)} {m2.group(1)}"
    m3 = re.match(r"^(.*), (The|A|An) \((.*)\)$", title)
    if m3:
        title = f"{m3.group(2)} {m3.group(1)} ({m3.group(3)})"
    return title, year


class Recommender:
    def __init__(self):
        import pandas as pd
        t0 = time.time()
        self.tr = data.load_train_csr(DATASET)
        self.n_users, self.n_items = self.tr.shape
        _, self.item_ids = data.load_id_maps(DATASET)
        movies = pd.read_csv(ROOT / "data" / "raw" / "movielens" / "ml-25m" / "movies.csv").set_index("movieId")
        self.titles, self.years, self.genres = [], [], []
        for mid in self.item_ids:
            row = movies.loc[int(mid)]
            t, y = display_title(row["title"])
            self.titles.append(t)
            self.years.append(y)
            self.genres.append([] if row["genres"] == "(no genres listed)" else row["genres"].split("|"))
        self.counts = np.asarray(self.tr.getnnz(axis=0)).astype(np.float64)
        self.mean_rating = np.asarray(self.tr.sum(axis=0)).ravel() / np.maximum(self.counts, 1)
        self.movie_to_idx = {int(m): n for n, m in enumerate(self.item_ids)}
        self.search_keys = [t.lower() for t in self.titles]
        self.by_popularity = np.argsort(-self.counts, kind="stable")

        self.item_knn = ItemKNN(k=ITEM_K, shrink=25.0).fit(
            self.tr, cache=str(processed_dir(DATASET) / "itemknn_sim_shrink25.npz"))
        self.user_knn = UserKNN(k=USER_K, shrink=25.0).fit(self.tr)
        z = np.load(processed_dir(DATASET) / "serving" / "mf.npz")
        self.mf = SimpleNamespace(Q=z["Q"].astype(np.float64), bi=z["bi"], mu=float(z["mu"]), lam=float(z["lam"]),
                                  biased=bool(z["biased"]))
        z = np.load(processed_dir(DATASET) / "serving" / "occf.npz")
        self.occf = SimpleNamespace(Q=z["Q"], alpha=float(z["alpha"]), lam=float(z["lam"]))
        self.load_seconds = time.time() - t0

    # ---------------------------------------------------------------- catalogue
    def card(self, i: int) -> dict:
        return {"movieId": int(self.item_ids[i]), "title": self.titles[i], "year": self.years[i],
                "genres": self.genres[i], "nRatings": int(self.counts[i]),
                "avgRating": round(float(self.mean_rating[i]), 2)}

    def search(self, q: str, limit: int = 12) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        hit = [n for n, t in enumerate(self.search_keys) if q in t]
        hit.sort(key=lambda n: (not self.search_keys[n].startswith(q), -self.counts[n]))
        return [self.card(n) for n in hit[:limit]]

    def popular(self, limit: int = 24, genre: str | None = None) -> list[dict]:
        out = []
        for n in self.by_popularity:
            if genre and genre not in self.genres[n]:
                continue
            out.append(self.card(int(n)))
            if len(out) >= limit:
                break
        return out

    def find_exact(self, title: str, year: int) -> int | None:
        t = title.lower()
        for n in np.nonzero([k == t for k in self.search_keys])[0]:
            if self.years[n] == year:
                return int(n)
        return None

    def genre_list(self) -> list[str]:
        return sorted({g for gs in self.genres for g in gs})

    # ---------------------------------------------------------- recommendations
    def _target(self, ratings: dict[int, float]) -> sp.csr_matrix:
        idx = np.array(list(ratings.keys()), dtype=np.int64)
        val = np.array(list(ratings.values()), dtype=np.float32)
        return sp.csr_matrix((val, (np.zeros(len(idx), dtype=np.int64), idx)), shape=(1, self.n_items))

    def _scores(self, model: str, target: sp.csr_matrix) -> np.ndarray:
        u = np.array([0])
        if model == "popularity":
            return self.counts.copy()
        if model == "item_cf":
            return np.asarray(self.item_knn.with_targets(target).score_fn(u)[0], dtype=np.float64)
        if model == "user_cf":
            c = self.user_knn.with_targets(target)
            c.exclude_self = False
            return np.asarray(c.score_fn(u)[0], dtype=np.float64)
        if model == "mf":
            P, bu = foldin_mf(self.mf, target, u)
            return P[0].astype(np.float64) @ self.mf.Q.T + self.mf.bi
        if model == "one_class":
            p = foldin_occf(self.occf, target, u)
            return p[0] @ self.occf.Q.T
        raise ValueError(model)

    def _because(self, items: np.ndarray, liked: dict[int, float]) -> dict[int, dict]:
        """For each recommended movie, the liked movie of the user most similar to it (item-item cosine)."""
        out = {}
        if not liked:
            return out
        sk = self.item_knn.sk
        for i in items:
            row = sk.getrow(int(i))
            best, best_s = None, 0.0
            for j, s in zip(row.indices, row.data):
                if int(j) in liked and s > best_s:
                    best, best_s = int(j), float(s)
            if best is not None:
                out[int(i)] = {"movieId": int(self.item_ids[best]), "title": self.titles[best],
                               "similarity": round(best_s, 3)}
        return out

    def recommend(self, ratings_by_movie: dict[int, float], models: list[str], k: int = 10,
                  genre: str | None = None) -> dict:
        ratings = {self.movie_to_idx[m]: float(r) for m, r in ratings_by_movie.items() if m in self.movie_to_idx}
        target = self._target(ratings)
        liked = {i: r for i, r in ratings.items() if r >= 4.0}
        seen = np.zeros(self.n_items, dtype=bool)
        seen[list(ratings.keys())] = True
        allowed = ~seen
        if genre:
            allowed &= np.array([genre in g for g in self.genres])
        out = {}
        for m in models:
            t0 = time.time()
            s = self._scores(m, target)
            fallback = False
            if m != "popularity" and not (np.max(s) > np.min(s)):
                s, fallback = self.counts.copy(), True
            s = np.where(allowed, s, -np.inf)
            kk = min(k, int(allowed.sum()))
            top = np.argpartition(-s, kk - 1)[:kk]
            top = top[np.argsort(-s[top], kind="stable")]
            because = self._because(top, liked)
            items = []
            for r, i in enumerate(top, 1):
                c = self.card(int(i))
                c.update(rank=r, score=round(float(s[i]), 4), because=because.get(int(i)))
                items.append(c)
            out[m] = {"label": MODELS[m], "fallback": fallback, "items": items,
                      "ms": round((time.time() - t0) * 1000)}
        return {"nRated": len(ratings), "nLiked": len(liked), "models": out}
