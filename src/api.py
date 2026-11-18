"""JSON API for the React web app (frontend/).

Run:  uvicorn src.api:app --port 8000
Result endpoints read the tables already produced by the pipeline (reports/tables/*.csv); nothing is recomputed.
The recommendation endpoints need the exported models (python -m src.export_models) and load in the background,
so /api/status reports when live recommendations are ready.
"""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
PROC = ROOT / "data" / "processed"
DATASETS = {"ml_25m": "MovieLens 25M", "netflix": "Netflix Prize (50,000-user sample)"}

@asynccontextmanager
async def lifespan(_app):
    _state["loading"] = True
    threading.Thread(target=_load_recommender, daemon=True).start()     # models load in the background
    yield


app = FastAPI(title="Movie recommender API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
if FIGURES.exists():
    app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")

_state: dict = {"rec": None, "error": None, "loading": False}


def _load_recommender() -> None:
    try:
        from src.serving import Recommender
        _state["rec"] = Recommender()
    except Exception as e:                       # noqa: BLE001 - reported through /api/status
        _state["error"] = f"{type(e).__name__}: {e}"
    finally:
        _state["loading"] = False


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records"))      # NaN -> null


def _table(name: str, dataset: str | None = None) -> pd.DataFrame:
    f = TABLES / f"{name}.csv"
    if not f.exists():
        raise HTTPException(404, f"{name}.csv not generated yet")
    df = pd.read_csv(f)
    if dataset is not None and "dataset" in df.columns:
        if dataset not in DATASETS:
            raise HTTPException(400, "unknown dataset")
        df = df[df["dataset"] == dataset]
    return df


@app.get("/api/status")
def status() -> dict:
    rec = _state["rec"]
    return {"ready": rec is not None, "loading": _state["loading"], "error": _state["error"],
            "loadSeconds": round(rec.load_seconds, 1) if rec else None}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    out = []
    for key, name in DATASETS.items():
        meta = json.loads((PROC / key / "meta.json").read_text())
        out.append({"key": key, "name": name, "users": meta["users"], "items": meta["items"],
                    "ratings": meta["ratings"], "density": meta["density"], "split": meta["split_sizes"]})
    return out


@app.get("/api/overview")
def overview(dataset: str = "ml_25m") -> dict:
    paired = _table("paired_comparisons", dataset)
    return {"main": _records(_table("main_results", dataset)), "ci": _records(_table("bootstrap_ci", dataset)),
            "seeds": _records(_table("seed_summary", dataset)), "paired": _records(paired)}


@app.get("/api/analysis/{kind}")
def analysis(kind: str, dataset: str = "ml_25m") -> list[dict]:
    names = {"history": "history_buckets", "coldstart": f"cold_start_{dataset}",
             "diagnostics": f"list_diagnostics_{dataset}", "curves": f"learning_curves_{dataset}"}
    if kind not in names:
        raise HTTPException(404, "unknown analysis")
    if dataset not in DATASETS:
        raise HTTPException(400, "unknown dataset")
    return _records(_table(names[kind], dataset if kind == "history" else None))


@app.get("/api/figures")
def figures() -> list[dict]:
    caps = {
        "fig_main_results.png": "Method comparison with 95% bootstrap intervals",
        "fig_tradeoff.png": "Rating accuracy vs ranking quality",
        "fig_sensitivity_cf_mf.png": "Hyperparameter sensitivity: CF and MF",
        "fig_sensitivity_oneclass.png": "Hyperparameter sensitivity: one-class CF",
        "fig_learning_curves.png": "Learning curves of the final models",
        "history_buckets.png": "Metrics by size of the user's history",
        "cold_start.png": "Simulated cold start",
        "list_diagnostics.png": "Coverage, popularity and diversity of recommended lists",
    }
    return [{"file": f, "url": f"/figures/{f}", "caption": c} for f, c in caps.items() if (FIGURES / f).exists()]


# ------------------------------------------------------------------ live recommendations
def _rec():
    rec = _state["rec"]
    if rec is None:
        raise HTTPException(503, _state["error"] or "Models are still loading, try again in a moment.")
    return rec


PRESETS = [
    {"id": "scifi", "name": "Sci-fi & action fan", "blurb": "Blockbusters from the 80s to today.",
     "ratings": [("The Matrix", 1999, 5), ("Inception", 2010, 5), ("Blade Runner", 1982, 4.5),
                 ("Terminator 2: Judgment Day", 1991, 5), ("Alien", 1979, 4.5), ("Interstellar", 2014, 4.5),
                 ("Avatar", 2009, 3), ("Star Wars: Episode IV - A New Hope", 1977, 5)]},
    {"id": "family", "name": "Family & animation", "blurb": "Animated favourites for all ages.",
     "ratings": [("Toy Story", 1995, 5), ("Finding Nemo", 2003, 5), ("The Lion King", 1994, 5),
                 ("Shrek", 2001, 4.5), ("Up", 2009, 4.5), ("Monsters, Inc.", 2001, 4.5),
                 ("Aladdin", 1992, 4), ("Frozen", 2013, 4)]},
    {"id": "classics", "name": "Classic drama & crime", "blurb": "Acclaimed films of the 40s to 90s.",
     "ratings": [("The Godfather", 1972, 5), ("Casablanca", 1942, 4.5), ("12 Angry Men", 1957, 5),
                 ("Goodfellas", 1990, 4.5), ("Taxi Driver", 1976, 4), ("Schindler's List", 1993, 5),
                 ("Citizen Kane", 1941, 4), ("Pulp Fiction", 1994, 4)]},
    {"id": "cold", "name": "Brand-new user (2 ratings)", "blurb": "See how each model copes with almost no history.",
     "ratings": [("Pulp Fiction", 1994, 5), ("Fight Club", 1999, 4)]},
]


@app.get("/api/movies/search")
def search(q: str = Query("", max_length=80), limit: int = Query(12, ge=1, le=30)) -> list[dict]:
    return _rec().search(q, limit)


@app.get("/api/movies/popular")
def popular(limit: int = Query(24, ge=1, le=60), genre: str | None = None) -> list[dict]:
    return _rec().popular(limit, genre)


@app.get("/api/genres")
def genres() -> list[str]:
    return _rec().genre_list()


@app.get("/api/presets")
def presets() -> list[dict]:
    rec = _rec()
    out = []
    for p in PRESETS:
        rows = []
        for title, year, r in p["ratings"]:
            i = rec.find_exact(title, year)
            if i is not None:
                rows.append({**rec.card(i), "rating": r})
        out.append({"id": p["id"], "name": p["name"], "blurb": p["blurb"], "ratings": rows})
    return out


class Rating(BaseModel):
    movieId: int
    rating: float = Field(ge=0.5, le=5.0)


class RecommendRequest(BaseModel):
    ratings: list[Rating] = Field(min_length=1, max_length=300)
    models: list[str] = ["popularity", "item_cf", "user_cf", "mf", "one_class"]
    k: int = Field(10, ge=1, le=30)
    genre: str | None = None


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict:
    rec = _rec()
    from src.serving import MODELS
    bad = [m for m in req.models if m not in MODELS]
    if bad:
        raise HTTPException(400, f"unknown model(s): {bad}")
    ratings = {r.movieId: r.rating for r in req.ratings}
    if not any(m in rec.movie_to_idx for m in ratings):
        raise HTTPException(400, "none of the rated movies are in the catalogue")
    return rec.recommend(ratings, req.models, req.k, req.genre)
