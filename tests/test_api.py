"""API and serving tests. Result endpoints need reports/tables; recommendation tests need the exported models
(python -m src.export_models) and are skipped otherwise."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import api
from src.config import processed_dir
from src.serving import display_title

HAVE_MODELS = (processed_dir("ml_25m") / "serving" / "occf.npz").exists()


def test_display_title():
    assert display_title("Matrix, The (1999)") == ("The Matrix", 1999)
    assert display_title("Toy Story (1995)") == ("Toy Story", 1995)
    assert display_title("American in Paris, An (1951)") == ("An American in Paris", 1951)
    assert display_title("No year") == ("No year", None)


def test_result_endpoints():
    c = TestClient(api.app)            # no startup context: models are not loaded here
    r = c.get("/api/overview?dataset=netflix")
    assert r.status_code == 200 and {"main", "ci", "seeds", "paired"} <= set(r.json())
    assert c.get("/api/analysis/history?dataset=ml_25m").json()
    assert c.get("/api/analysis/nope").status_code == 404
    assert c.get("/api/overview?dataset=bad").status_code == 400
    assert c.post("/api/recommend", json={"ratings": [{"movieId": 1, "rating": 5}]}).status_code == 503


@pytest.mark.skipif(not HAVE_MODELS, reason="exported models not available")
def test_recommend_live():
    with TestClient(api.app) as c:
        for _ in range(120):
            if c.get("/api/status").json()["ready"]:
                break
            import time; time.sleep(1)
        assert c.get("/api/status").json()["ready"]
        presets = c.get("/api/presets").json()
        p = next(x for x in presets if x["id"] == "scifi")
        body = {"ratings": [{"movieId": r["movieId"], "rating": r["rating"]} for r in p["ratings"]], "k": 7}
        out = c.post("/api/recommend", json=body).json()
        rated = {r["movieId"] for r in body["ratings"]}
        assert set(out["models"]) == {"popularity", "item_cf", "user_cf", "mf", "one_class"}
        for m, v in out["models"].items():
            ids = [i["movieId"] for i in v["items"]]
            assert len(ids) == 7 and len(set(ids)) == 7 and not rated & set(ids)      # k items, no repeats, nothing already rated
        # a single 5-star rating gives CF no centred signal -> popularity fallback is flagged
        one = c.post("/api/recommend", json={"ratings": [body["ratings"][0]], "models": ["item_cf", "popularity"]}).json()
        assert one["models"]["item_cf"]["fallback"] and not one["models"]["popularity"]["fallback"]
        assert c.post("/api/recommend", json={"ratings": [{"movieId": 1, "rating": 9}]}).status_code == 422
