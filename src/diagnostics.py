"""Beyond-accuracy diagnostics of the recommended lists (task 6.5): WHY do the methods differ?

Each method's frozen ranking configuration is refit, then its top-10 list (already-rated items excluded, same
protocol as the evaluation) is generated for a fixed random sample of eligible users (--users, default 10,000;
the same users for every method). For the lists we report
  coverage        distinct items recommended / catalogue size (items in the model)
  gini            inequality of how often items are recommended (0 = all equally, 1 = one item always)
  pop_pct         mean popularity percentile of recommended items (0 = the most rated item, 1 = the least)
  median_ratings  median number of TRAIN ratings of a recommended item
  head_share      share of recommended slots filled by the 1% most-rated items
  tail_share      share filled by items in the less-rated half of the catalogue
  novelty         mean self-information -log2(count_i / total) of recommended items (higher = more obscure)
  item_mean       mean of the recommended items' average train rating (quality as the crowd rated it)
  genres          (MovieLens) mean number of distinct genres in a top-10 list
plus NDCG@10 of the same sampled users for reference.

Outputs: reports/tables/list_diagnostics.{csv,md}, reports/figures/list_diagnostics.png
Run:  python -m src.diagnostics --dataset ml_25m      (then --report)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import data
from src.config import SEED
from src.eval import _topk
from src.final_run import fit_key, fit_model, job_key, set_k
from src.final_tables import DISPLAY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
K = 10


def top_lists(model, users, exclude, k: int = K, batch: int = 2000) -> np.ndarray:
    out = []
    for s in range(0, len(users), batch):
        b = users[s:s + batch]
        sc = np.array(model.score_fn(b), dtype=np.float32)
        seen = exclude[b].tocoo()
        sc[seen.row, seen.col] = -np.inf
        out.append(_topk(sc, k))
    return np.vstack(out)


def gini(counts: np.ndarray) -> float:
    c = np.sort(counts.astype(float))
    n, total = len(c), c.sum()
    if total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * total))


def list_metrics(L: np.ndarray, item_count: np.ndarray, item_mean: np.ndarray, genres: list | None = None) -> dict:
    """Diagnostics of an (n_users, K) array of recommended item indices."""
    n_items = len(item_count)
    # popularity percentile: 0 = most rated item, 1 = least rated (ties share the better percentile)
    order = np.argsort(-item_count, kind="stable")
    pop_pct = np.empty(n_items)
    pop_pct[order] = np.arange(n_items) / n_items
    flat = L.ravel()
    rec_counts = np.bincount(flat, minlength=n_items)
    p = item_count / item_count.sum()
    out = dict(
        coverage=float((rec_counts > 0).sum() / n_items),
        gini=gini(rec_counts),
        pop_pct=float(pop_pct[flat].mean()),
        median_ratings=float(np.median(item_count[flat])),
        head_share=float((pop_pct[flat] < 0.01).mean()),
        tail_share=float((pop_pct[flat] >= 0.5).mean()),
        novelty=float(-np.log2(np.maximum(p[flat], 1e-12)).mean()),
        item_mean=float(item_mean[flat].mean()),
    )
    if genres is not None:
        out["genres"] = float(np.mean([len(set().union(*(genres[i] for i in row))) for row in L]))
    return out


def load_genres(dataset: str):
    if not dataset.startswith("ml"):
        return None
    movies = pd.read_csv(ROOT / "data" / "raw" / "movielens" / "ml-25m" / "movies.csv")
    g = dict(zip(movies.movieId, movies.genres.str.split("|")))
    _, item_ids = data.load_id_maps(dataset)
    return [set(x for x in g.get(int(m), []) if x != "(no genres listed)") for m in item_ids]


def run(dataset: str, n_users: int, config_path: str) -> pd.DataFrame:
    methods = json.loads(Path(config_path).read_text())["datasets"][dataset]
    train, val, test_full = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    nu, ni = tr.shape
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)
    eligible = np.sort(test_full.loc[test_full["rating"] >= 4.0, "u"].unique())
    users = np.sort(np.random.default_rng(SEED).choice(eligible, min(n_users, len(eligible)), replace=False))
    test = test_full[test_full["u"].isin(users)]
    count = np.bincount(train["i"], minlength=ni).astype(float)
    mean = np.bincount(train["i"], weights=train["rating"], minlength=ni) / np.maximum(count, 1)
    genres = load_genres(dataset)
    ctx = dict(dataset=dataset, train=train, val=val, n_users=nu, n_items=ni, tr=tr)
    print(f"{dataset}: {len(users):,} users, catalogue {ni:,}", flush=True)

    jobs: dict[str, dict] = {}
    for label, spec in methods.items():
        if "ranking" in spec:
            j = jobs.setdefault(job_key(spec["ranking"]), dict(spec=spec["ranking"], labels=[]))
            j["labels"].append(label)
    rows, model, mkey = [], None, None
    for j in sorted(jobs.values(), key=lambda j: fit_key(j["spec"])):
        spec, t0 = j["spec"], time.time()
        if fit_key(spec) != mkey:
            model = None
            model = fit_model(spec, ctx)
            mkey = fit_key(spec)
        set_k(model, spec, "ranking")
        L = top_lists(model, users, trval)
        m = list_metrics(L, count, mean, genres)
        # NDCG@10 of exactly these users, from the lists
        rel = test[test["rating"] >= 4.0]
        relset = {u: set(g["i"]) for u, g in rel.groupby("u")}
        disc = 1 / np.log2(np.arange(2, K + 2))
        nds = []
        for u, row in zip(users, L):
            r = relset.get(u, set())
            if not r:
                continue
            hits = np.array([i in r for i in row], dtype=float)
            nds.append((hits * disc).sum() / disc[:min(len(r), K)].sum())
        m["ndcg@10_sample"] = float(np.mean(nds))
        for label in j["labels"]:
            rows.append(dict(dataset=dataset, method=label, users=len(users), **m))
        print(f"  {j['labels'][0]:<28} cov={m['coverage']:.3f} pop_pct={m['pop_pct']:.3f} "
              f"median_n={m['median_ratings']:.0f} tail={m['tail_share']:.3f} ndcg={m['ndcg@10_sample']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"list_diagnostics_{dataset}.csv", index=False)
    return df


COLS = [("coverage", "Coverage", "{:.3f}"), ("gini", "Gini", "{:.3f}"), ("pop_pct", "Popularity pct.", "{:.3f}"),
        ("median_ratings", "Median #ratings", "{:,.0f}"), ("head_share", "Top-1% share", "{:.3f}"),
        ("tail_share", "Less-rated-half share", "{:.3f}"), ("novelty", "Novelty (bits)", "{:.2f}"),
        ("item_mean", "Avg item rating", "{:.2f}"), ("genres", "Genres/list", "{:.2f}"),
        ("ndcg@10_sample", "NDCG@10", "{:.4f}")]


def report() -> None:
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("list_diagnostics_*.csv"))], ignore_index=True)
    df.to_csv(OUT / "list_diagnostics.csv", index=False)
    md = ["Top-10 list diagnostics (seed 42; frozen ranking configurations; fixed random sample of eligible users "
          "per dataset; already-rated items excluded). Popularity percentile: 0 = most-rated item, 1 = least-rated; "
          "'Less-rated-half' = items below the median in number of training ratings.\n"]
    for ds, g in df.groupby("dataset", sort=False):
        cols = [c for c in COLS if c[0] in g and g[c[0]].notna().any()]
        md.append(f"**{DISPLAY.get(ds, ds)}** ({int(g['users'].iloc[0]):,} users)\n")
        md.append("| Method | " + " | ".join(c[1] for c in cols) + " |")
        md.append("|:--|" + "---:|" * len(cols))
        for _, r in g.iterrows():
            md.append(f"| {r['method']} | " + " | ".join(c[2].format(r[c[0]]) for c in cols) + " |")
        md.append("")
    (OUT / "list_diagnostics.md").write_text("\n".join(md))

    datasets = list(dict.fromkeys(df.dataset))
    panels = [("coverage", "Item coverage"), ("median_ratings", "Median #train ratings of recommended items"),
              ("tail_share", "Share of recs from less-rated half")]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 3.6 * len(datasets)), squeeze=False)
    colors = {"Popularity (most rated)": "#888888", "Mean rating (min 50 ratings)": "#bbbbbb",
              "Item-based CF": "#1f77b4", "User-based CF": "#17becf", "MF (biased)": "#d62728",
              "MF (plain)": "#ff9896", "One-class WRMF": "#2ca02c", "One-class dual-regularized": "#006400"}
    for i, ds in enumerate(datasets):
        g = df[df.dataset == ds]
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            ax.barh(g.method, g[col], color=[colors.get(m, "k") for m in g.method])
            if col == "median_ratings":
                ax.set_xscale("log")
            ax.invert_yaxis()
            ax.set_title(f"{DISPLAY.get(ds, ds)}\n{title}", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            if j > 0:
                ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(FIG / "list_diagnostics.png", dpi=160)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--users", type=int, default=10000)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.dataset, a.users, a.config)
