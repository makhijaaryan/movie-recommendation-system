"""History-bucket analysis (task 6.3, research question 5): how does each method behave for users with
little vs. much rating history?

Uses the seed-42 per-user results saved by `python -m src.final_run`; users are grouped by their number of
TRAIN ratings (<30, 30-59, 60-119, 120-299, 300+; see src/eval.py). Every user in the data has at least 20
ratings, so this is *natural* variation in history length; the simulated cold start (down to 1 rating) is task 6.4.

Outputs in reports/tables/: history_buckets.{csv,md}; figure reports/figures/history_buckets.png
Run:  python -m src.buckets
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
from src.final_tables import DISPLAY
from src.uncertainty import boot_sums, ci, load_per_user

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
B = 500


def bucket_of(dataset: str) -> pd.Series:
    """Bucket label per user index (by number of train ratings)."""
    counts = np.diff(data.load_train_csr(dataset).indptr)
    return pd.Series(pd.cut(counts, bins=list(BUCKET_EDGES), labels=list(BUCKET_LABELS), right=False),
                     index=np.arange(len(counts)), name="bucket"), counts


def analyse(dataset: str, methods: dict) -> pd.DataFrame:
    bk, counts = bucket_of(dataset)
    rows = []
    for label, spec in methods.items():
        if "ranking" in spec:
            f = load_per_user(dataset, label, "ranking").set_index("u")
            f["bucket"] = bk.loc[f.index].to_numpy()
            for b, g in f.groupby("bucket", observed=True):
                X = np.column_stack([g[["ndcg@10", "hit@10", "recall@10"]].to_numpy(), np.ones(len(g))])
                s = boot_sums(X, B, seed=2)
                for j, m in enumerate(("ndcg@10", "hit@10", "recall@10")):
                    lo, hi = ci(s[:, j] / s[:, 3])
                    rows.append(dict(dataset=dataset, method=label, bucket=str(b), metric=m,
                                     value=float(g[m].mean()), ci_low=lo, ci_high=hi, n_users=len(g)))
        if "rating" in spec:
            f = load_per_user(dataset, label, "rating").set_index("u")
            f["bucket"] = bk.loc[f.index].to_numpy()
            for b, g in f.groupby("bucket", observed=True):
                s = boot_sums(g[["sse", "n"]].to_numpy(float), B, seed=3)
                lo, hi = ci(np.sqrt(s[:, 0] / s[:, 1]))
                rows.append(dict(dataset=dataset, method=label, bucket=str(b), metric="rmse",
                                 value=float(np.sqrt(g.sse.sum() / g.n.sum())), ci_low=lo, ci_high=hi,
                                 n_users=len(g)))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, metric: str, title: str, d: int = 4) -> str:
    out = []
    for dataset, g in df[df.metric == metric].groupby("dataset", sort=False):
        sizes = g.drop_duplicates("bucket").set_index("bucket")["n_users"]
        # users with ranking metrics and with rating metrics differ in number; use the largest count shown
        out.append(f"**{DISPLAY.get(dataset, dataset)} — {title}**\n")
        out.append("| Method | " + " | ".join(f"{b} (n={int(sizes[b]):,})" for b in BUCKET_LABELS if b in sizes.index) + " |")
        out.append("|:--|" + "---:|" * len(sizes))
        for label in dict.fromkeys(g.method):
            r = g[g.method == label].set_index("bucket")
            out.append(f"| {label} | " + " | ".join(f"{r.loc[b, 'value']:.{d}f}" if b in r.index else "–"
                                                    for b in BUCKET_LABELS if b in sizes.index) + " |")
        out.append("")
    return "\n".join(out)


def relative_to_popularity(df: pd.DataFrame, metric: str = "ndcg@10") -> str:
    """NDCG@10 divided by the popularity baseline's NDCG@10 in the same bucket. Absolute ranking metrics
    grow with history for every method (heavy users have more held-out relevant movies), so this ratio
    isolates how much a method adds over 'recommend what everybody watches' at each history size."""
    pop = "Popularity (most rated)"
    out = []
    for dataset, g in df[df.metric == metric].groupby("dataset", sort=False):
        t = g.pivot(index="method", columns="bucket", values="value")[list(BUCKET_LABELS)]
        rel = t.div(t.loc[pop], axis=1).drop(index=pop)
        out.append(f"**{DISPLAY.get(dataset, dataset)} — {metric.upper()} relative to popularity (1.00 = same as popularity)**\n")
        out.append("| Method | " + " | ".join(BUCKET_LABELS) + " |")
        out.append("|:--|" + "---:|" * len(BUCKET_LABELS))
        for label, r in rel.iterrows():
            out.append(f"| {label} | " + " | ".join(f"{v:.2f}" for v in r) + " |")
        out.append("")
    return "\n".join(out)


def plot(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    datasets = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.6 * len(datasets), 7.6), sharex=True)
    style = {"Popularity (most rated)": ("#888888", "--"), "Mean rating (min 50 ratings)": ("#bbbbbb", ":"),
             "Item-based CF": ("#1f77b4", "-"), "User-based CF": ("#17becf", "-"),
             "MF (biased)": ("#d62728", "-"), "MF (plain)": ("#ff9896", "-"),
             "One-class WRMF": ("#2ca02c", "-"), "One-class dual-regularized": ("#006400", "-"),
             "Global mean": ("#dddddd", ":")}
    for j, ds in enumerate(datasets):
        for i, (metric, ylabel) in enumerate((("ndcg@10", "NDCG@10 (higher is better)"),
                                              ("rmse", "RMSE (lower is better)"))):
            ax = axes[i, j]
            g = df[(df.dataset == ds) & (df.metric == metric)]
            for label in dict.fromkeys(g.method):
                r = g[g.method == label].set_index("bucket").reindex(BUCKET_LABELS).dropna(subset=["value"])
                c, ls = style.get(label, ("k", "-"))
                x = [BUCKET_LABELS.index(b) for b in r.index]
                ax.plot(x, r.value, ls, color=c, marker="o", ms=4, label=label)
                ax.fill_between(x, r.ci_low, r.ci_high, color=c, alpha=0.15, lw=0)
            ax.set_xticks(range(len(BUCKET_LABELS)), BUCKET_LABELS)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(DISPLAY.get(ds, ds))
            if i == 1:
                ax.set_xlabel("number of training ratings of the user")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "history_buckets.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    df = pd.concat([analyse(ds, m) for ds, m in config["datasets"].items()], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "history_buckets.csv", index=False)
    md = ("Test metrics by the number of the user's training ratings (seed 42; 95% bootstrap intervals are in "
          "`history_buckets.csv`). Ranking metrics at K = 10; n = users with ranking metrics (RMSE tables use all "
          "users).\n\n" + to_markdown(df, "ndcg@10", "NDCG@10") + "\n" + to_markdown(df, "hit@10", "Hit@10")
          + "\n" + to_markdown(df, "rmse", "RMSE") + "\n" + relative_to_popularity(df))
    (OUT / "history_buckets.md").write_text(md)
    plot(df)
    print(md)


if __name__ == "__main__":
    main()
