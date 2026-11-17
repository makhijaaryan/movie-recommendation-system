"""Uncertainty analysis for the final results (task 6.2).

1. Bootstrap (users are resampled with replacement, B = 1000, fixed RNG): 95% percentile confidence
   intervals for every method's test metrics and for the PAIRED difference of the comparisons listed in
   experiments/final_config.json. Resampling the same users for both methods keeps the pairing, so a
   difference whose interval excludes 0 is a real difference for this user population.
   Needs the per-user files written by `python -m src.final_run` for seed 42.
2. Seed summary: mean +/- standard deviation of the test metrics over the seeds logged in
   experiments/results.csv (each seed = a new data split / Netflix user sample / model initialisation,
   same frozen hyperparameters).

Outputs in reports/tables/: bootstrap_ci.{csv,md}, paired_comparisons.{csv,md}, seed_summary.{csv,md}
Run:  python -m src.uncertainty [--no-bootstrap] [--no-seeds]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import processed_dir
from src.final_run import RANK_COLS, slug
from src.final_tables import DISPLAY, RESULTS, build

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"
B_DEFAULT = 1000
RATING_METRICS = ("rmse", "mae")


# ----------------------------------------------------------------------------- bootstrap
def boot_sums(X: np.ndarray, B: int = B_DEFAULT, seed: int = 0, chunk: int = 50) -> np.ndarray:
    """Weighted column sums of X (n, m) under B bootstrap resamples of the n rows -> (B, m).

    A resample is a multinomial count vector over rows; the same seed gives the same resamples, so
    two metrics (or two methods) computed from the same rows are paired."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    p = np.full(n, 1.0 / n)
    out = []
    for s in range(0, B, chunk):
        W = rng.multinomial(n, p, size=min(chunk, B - s)).astype(np.float32)
        out.append(W @ X.astype(np.float32))
    return np.concatenate(out).astype(np.float64)


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def load_per_user(dataset: str, label: str, role: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(dataset) / "per_user" / f"{slug(label)}__{role}.parquet")


def bootstrap_dataset(dataset: str, methods: dict, B: int = B_DEFAULT):
    """Returns (point, reps): point[(label, metric)] = estimate, reps[(label, metric)] = (B,) array."""
    point, reps = {}, {}
    # --- ranking metrics: same users for every method (verified), one shared set of resamples
    rank_frames = {l: load_per_user(dataset, l, "ranking").set_index("u").sort_index()
                   for l, s in methods.items() if "ranking" in s}
    users = None
    for f in rank_frames.values():
        users = f.index if users is None else users.intersection(f.index)
    sizes = {len(f) for f in rank_frames.values()}
    if len(sizes) != 1 or len(users) != next(iter(sizes)):
        print(f"  warning: ranking user sets differ across methods ({sizes}); using the {len(users)} common users")
    for label, f in rank_frames.items():
        f = f.loc[users]
        X = np.column_stack([f[RANK_COLS].to_numpy(), np.ones(len(f))])
        sums = boot_sums(X, B)
        for j, c in enumerate(RANK_COLS):
            point[(label, c)] = float(f[c].mean())
            reps[(label, c)] = sums[:, j] / sums[:, -1]
    # --- rating metrics: all users with test ratings
    rat_frames = {l: load_per_user(dataset, l, "rating").set_index("u").sort_index()
                  for l, s in methods.items() if "rating" in s}
    rusers = None
    for f in rat_frames.values():
        rusers = f.index if rusers is None else rusers.intersection(f.index)
    for label, f in rat_frames.items():
        f = f.loc[rusers]
        sums = boot_sums(f[["sse", "sae", "n"]].to_numpy(float), B, seed=1)
        point[(label, "rmse")] = float(np.sqrt(f.sse.sum() / f.n.sum()))
        point[(label, "mae")] = float(f.sae.sum() / f.n.sum())
        reps[(label, "rmse")] = np.sqrt(sums[:, 0] / sums[:, 2])
        reps[(label, "mae")] = sums[:, 1] / sums[:, 2]
    return point, reps, len(users), len(rusers)


def fmt_ci(p: float, lo: float, hi: float, d: int = 4) -> str:
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def run_bootstrap(config: dict, B: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ci_rows, cmp_rows = [], []
    for dataset, methods in config["datasets"].items():
        print(f"bootstrap {dataset} ...", flush=True)
        point, reps, n_rank, n_rate = bootstrap_dataset(dataset, methods, B)
        for (label, metric), p in point.items():
            lo, hi = ci(reps[(label, metric)])
            ci_rows.append(dict(dataset=dataset, method=label, metric=metric, estimate=p, ci_low=lo, ci_high=hi,
                                n_users=n_rate if metric in RATING_METRICS else n_rank))
        for c in config.get("comparisons", []):
            if c["dataset"] != dataset:
                continue
            for metric in c["metrics"]:
                ka, kb = (c["a"], metric), (c["b"], metric)
                if ka not in reps or kb not in reps:
                    continue
                diff = reps[ka] - reps[kb]
                lo, hi = ci(diff)
                cmp_rows.append(dict(dataset=dataset, a=c["a"], b=c["b"], metric=metric,
                                     a_value=point[ka], b_value=point[kb], diff=point[ka] - point[kb],
                                     ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    ci_df, cmp_df = pd.DataFrame(ci_rows), pd.DataFrame(cmp_rows)
    ci_df.to_csv(OUT / "bootstrap_ci.csv", index=False)
    cmp_df.to_csv(OUT / "paired_comparisons.csv", index=False)

    # compact markdown: RMSE, Hit@10 and NDCG@10 with intervals
    shown = ["rmse", "mae", "hit@10", "ndcg@10"]
    names = {"rmse": "RMSE", "mae": "MAE", "hit@10": "Hit@10", "ndcg@10": "NDCG@10"}
    md = []
    for dataset, g in ci_df.groupby("dataset", sort=False):
        md.append(f"**{DISPLAY.get(dataset, dataset)}** — point estimate [95% bootstrap CI], {B} resamples of users\n")
        md.append("| Method | " + " | ".join(names[m] for m in shown) + " |")
        md.append("|:--|" + "---:|" * len(shown))
        for label in dict.fromkeys(g.method):
            cells = []
            for m in shown:
                r = g[(g.method == label) & (g.metric == m)]
                cells.append(fmt_ci(*r.iloc[0][["estimate", "ci_low", "ci_high"]]) if len(r) else "–")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "bootstrap_ci.md").write_text("\n".join(md))

    md = ["Paired differences (A − B); the same resampled users are used for both methods. "
          "'Real' means the 95% interval excludes 0. For RMSE a negative difference favours A.\n"]
    for dataset, g in cmp_df.groupby("dataset", sort=False):
        md.append(f"**{DISPLAY.get(dataset, dataset)}**\n")
        md.append("| A | B | Metric | A | B | A − B [95% CI] | Real? |")
        md.append("|:--|:--|:--|---:|---:|---:|:--:|")
        for _, r in g.iterrows():
            md.append(f"| {r.a} | {r.b} | {names.get(r.metric, r.metric)} | {r.a_value:.4f} | {r.b_value:.4f} | "
                      f"{fmt_ci(r['diff'], r.ci_low, r.ci_high)} | {'yes' if r.excludes_zero else 'no'} |")
        md.append("")
    (OUT / "paired_comparisons.md").write_text("\n".join(md))
    print((OUT / "paired_comparisons.md").read_text())


# ------------------------------------------------------------------------------- seeds
def run_seeds(config: dict) -> None:
    results = pd.read_csv(RESULTS)
    seeds = [s for s in config["seeds"] if (results.seed == s).any()]
    frames = []
    for s in seeds:
        d = build(results, config, seed=s)
        d["seed"] = s
        frames.append(d)
    allf = pd.concat(frames, ignore_index=True)
    metrics = ["rmse", "mae", "precision@10", "recall@10", "hit@10", "ndcg@10"]
    g = allf.groupby(["dataset", "method"], sort=False)[metrics]
    summary = pd.concat({"mean": g.mean(), "std": g.std(ddof=1), "n_seeds": g.count()["rmse"].combine_first(g.count()["ndcg@10"])},
                        axis=1)
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns]
    summary.reset_index().to_csv(OUT / "seed_summary.csv", index=False)
    allf.to_csv(OUT / "seed_runs.csv", index=False)

    names = {"rmse": "RMSE", "mae": "MAE", "precision@10": "P@10", "recall@10": "R@10", "hit@10": "Hit@10",
             "ndcg@10": "NDCG@10"}
    md = [f"Mean ± standard deviation over seeds {seeds} (new split, user sample and initialisation each time; "
          "hyperparameters frozen from seed 42).\n"]
    for dataset, gd in allf.groupby("dataset", sort=False):
        md.append(f"**{DISPLAY.get(dataset, dataset)}**\n")
        md.append("| Method | " + " | ".join(names[m] for m in metrics) + " |")
        md.append("|:--|" + "---:|" * len(metrics))
        for label in dict.fromkeys(gd.method):
            rows = gd[gd.method == label]
            cells = []
            for m in metrics:
                v = rows[m].dropna()
                cells.append("–" if v.empty else f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "seed_summary.md").write_text("\n".join(md))
    print((OUT / "seed_summary.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    if not a.no_bootstrap:
        run_bootstrap(config, a.B)
    if not a.no_seeds:
        run_seeds(config)


if __name__ == "__main__":
    main()
