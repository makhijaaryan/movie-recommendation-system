import numpy as np
import pandas as pd

from src.examples import MODELS, OC, POP, pick_cases


def test_pick_cases_follows_rules_and_is_deterministic():
    n = 400
    rng = np.random.default_rng(0)
    feat = pd.DataFrame({"n_train": rng.integers(20, 500, n), "niche": rng.random(n)})
    hits = {m: rng.integers(0, 6, n) for m in MODELS}
    n_rel = rng.integers(1, 20, n)
    a = pick_cases(feat, hits, n_rel, seed=1)
    assert a == pick_cases(feat, hits, n_rel, seed=1)
    u, cnt = a["S2 success, light user (<30 ratings), popularity misses"]
    if u is not None:
        assert feat.n_train[u] < 30 and hits[OC][u] >= 2 and hits[POP][u] == 0
    u, _ = a["F2b strict version: popularity >= 3 hits, one-class none"]
    if u is not None:
        assert hits[POP][u] >= 3 and hits[OC][u] == 0
    hits0 = {m: np.zeros(n, int) for m in MODELS}
    assert pick_cases(feat, hits0, n_rel)["S1 success, heavy user with mainstream taste"][0] is None   # empty rule -> None
