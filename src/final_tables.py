"""Build the main results table (the proposal's comparison table) from experiments/results.csv.

Which logged test row feeds each cell is declared in experiments/final_config.json:
RMSE/MAE come from a method's `rating` row, ranking metrics from its `ranking` row.
Each spec must match exactly one logged test row, otherwise the script stops, so the
table can never silently mix up configurations.

Outputs: reports/tables/main_results.csv  (all numbers, K = 5/10/20, source configs)
         reports/tables/main_results.md   (compact, K = 10, best value in bold)
         reports/tables/main_results_full.md  (K = 5, 10, 20)
and refreshes the table in README.md between its RESULTS-TABLE markers.

Run from the project root:  python -m src.final_tables
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.csv"
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"
README = ROOT / "README.md"
DISPLAY = {"ml_25m": "MovieLens 25M", "netflix": "Netflix Prize (50,000-user sample)"}
RANK_METRICS = ("precision", "recall", "hit", "ndcg")


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def select_row(results: pd.DataFrame, dataset: str, spec: dict) -> pd.Series:
    """The single TEST row of `dataset` whose model and params match `spec`; error if 0 or >1."""
    t = results[(results["split"] == "test") & (results["dataset"] == dataset) & (results["model"] == spec["model"])]
    want = spec.get("params", {})
    hits = []
    for _, row in t.iterrows():
        got = json.loads(row["params"])
        if all(k in got and _same(got[k], v) for k, v in want.items()):
            hits.append(row)
    if len(hits) != 1:
        raise ValueError(f"{dataset}/{spec}: expected exactly 1 matching test row, found {len(hits)}")
    return hits[0]


def build(results: pd.DataFrame, config: dict) -> pd.DataFrame:
    ks = config["ks"]
    rows = []
    for dataset, methods in config["datasets"].items():
        for label, spec in methods.items():
            out = {"dataset": dataset, "method": label}
            if "rating" in spec:
                r = select_row(results, dataset, spec["rating"])
                out.update(rmse=r["rmse"], mae=r["mae"], rating_config=r["params"])
            if "ranking" in spec:
                r = select_row(results, dataset, spec["ranking"])
                for k in ks:
                    for m in RANK_METRICS:
                        out[f"{m}@{k}"] = r[f"{m}@{k}"]
                out.update(n_users=int(r["n_users"]), ranking_config=r["params"])
            rows.append(out)
    return pd.DataFrame(rows)


def _fmt(v, best) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v:.4f}"
    return f"**{s}**" if best is not None and math.isclose(v, best, rel_tol=1e-12) else s


def to_markdown(df: pd.DataFrame, ks, lower_better=("rmse", "mae")) -> str:
    """One table per dataset, columns RMSE, MAE and the ranking metrics for the given Ks; best in bold."""
    cols = ["rmse", "mae"] + [f"{m}@{k}" for k in ks for m in RANK_METRICS]
    header = {"rmse": "RMSE", "mae": "MAE", "precision": "P", "recall": "R", "hit": "Hit", "ndcg": "NDCG"}
    names = [header[c] if c in header else f"{header[c.split('@')[0]]}@{c.split('@')[1]}" for c in cols]
    parts = []
    for dataset, g in df.groupby("dataset", sort=False):
        best = {}
        for c in cols:
            vals = g[c].dropna() if c in g else pd.Series(dtype=float)
            if len(vals):
                best[c] = vals.min() if c in lower_better else vals.max()
        n = int(g["n_users"].dropna().iloc[0]) if "n_users" in g and g["n_users"].notna().any() else None
        parts.append(f"**{DISPLAY.get(dataset, dataset)}**" + (f" — ranking metrics over {n:,} users" if n else "") + "\n")
        lines = ["| Method | " + " | ".join(names) + " |", "|:--|" + "---:|" * len(cols)]
        for _, r in g.iterrows():
            cells = [_fmt(r[c] if c in g else None, best.get(c)) for c in cols]
            lines.append(f"| {r['method']} | " + " | ".join(cells) + " |")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


NOTE = ("RMSE and MAE come from each method's configuration with the best *validation* RMSE; ranking metrics "
        "(K = 10 here; relevant = held-out rating ≥ 4; already-rated items excluded) come from its configuration "
        "with the best *validation* NDCG@10. Matrix factorization and one-class CF are trained for different "
        "objectives, hence two settings. One-class CF predicts no star ratings. Best value per column in bold. "
        "All numbers are single runs (seed 42) on the held-out test split.")


def main() -> None:
    results = pd.read_csv(RESULTS)
    config = json.loads(CONFIG.read_text())
    df = build(results, config)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "main_results.csv", index=False)
    compact = to_markdown(df, [10])
    full = to_markdown(df, config["ks"])
    (OUT / "main_results.md").write_text(compact + "\n" + NOTE + "\n")
    (OUT / "main_results_full.md").write_text(full + "\n" + NOTE + "\n")
    text = README.read_text()
    a, b = "<!-- RESULTS-TABLE-START -->", "<!-- RESULTS-TABLE-END -->"
    if a in text and b in text:
        head, rest = text.split(a, 1)
        _, tail = rest.split(b, 1)
        README.write_text(f"{head}{a}\n{compact}\n_{NOTE}_\n{b}{tail}")
    print(compact)


if __name__ == "__main__":
    main()
