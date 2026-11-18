"""Report figures and an index of every figure/table (task 6.7).

Built from files already produced by the pipeline; nothing is retrained here:
  fig_main_results.png        RMSE and NDCG@10 per method with 95% bootstrap intervals (bootstrap_ci.csv)
  fig_tradeoff.png            rating accuracy vs ranking quality: best RMSE is not best top-K (bootstrap_ci.csv)
  fig_sensitivity_cf_mf.png   validation RMSE / NDCG@10 vs neighbours k, MF latent dimension k and lambda (results.csv)
  fig_sensitivity_oneclass.png one-class CF validation NDCG@10 vs k, alpha, lambda, graph strength (results.csv)
  fig_learning_curves.png     MF train/validation RMSE per epoch, one-class validation NDCG per iteration (curves.py)
and reports/figures/INDEX.md describing all figures and tables available for the report.

Run:  python -m src.report_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_tables import DISPLAY

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.csv"
CONFIG = ROOT / "experiments" / "final_config.json"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
COLOR = {"Global mean": "#cccccc", "Popularity (most rated)": "#888888", "Mean rating (min 50 ratings)": "#bbbbbb",
         "Item-based CF": "#1f77b4", "User-based CF": "#17becf", "MF (biased)": "#d62728", "MF (plain)": "#ff9896",
         "One-class WRMF": "#2ca02c", "One-class dual-regularized": "#006400"}
DATASETS = ["ml_25m", "netflix"]


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------- main results
def fig_main_results(ci: pd.DataFrame, order: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for j, ds in enumerate(DATASETS):
        for i, (metric, title, lower) in enumerate((("rmse", "RMSE (lower is better)", True),
                                                    ("ndcg@10", "NDCG@10 (higher is better)", False))):
            ax = axes[i, j]
            g = ci[(ci.dataset == ds) & (ci.metric == metric)].set_index("method").reindex(order).dropna(subset=["estimate"])
            err = np.vstack([g.estimate - g.ci_low, g.ci_high - g.estimate])
            ax.barh(g.index, g.estimate, xerr=err, color=[COLOR[m] for m in g.index], ecolor="k", capsize=3)
            ax.invert_yaxis()
            if lower:
                ax.set_xlim(0.6, g.estimate.max() * 1.03)
            for y, (m, r) in enumerate(g.iterrows()):
                ax.text(r.estimate, y, f" {r.estimate:.3f}", va="center", fontsize=8)
            ax.set_title(f"{DISPLAY[ds]}\n{title}", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            if j == 1:
                ax.set_yticklabels([])
    fig.tight_layout()
    _save(fig, "fig_main_results.png")


def fig_tradeoff(ci: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, ds in zip(axes, DATASETS):
        g = ci[ci.dataset == ds]
        rm = g[g.metric == "rmse"].set_index("method")
        nd = g[g.metric == "ndcg@10"].set_index("method")
        for m in rm.index.intersection(nd.index):
            ax.errorbar(rm.loc[m, "estimate"], nd.loc[m, "estimate"], color=COLOR[m], marker="o", ms=9, label=m,
                        xerr=[[rm.loc[m, "estimate"] - rm.loc[m, "ci_low"]], [rm.loc[m, "ci_high"] - rm.loc[m, "estimate"]]],
                        yerr=[[nd.loc[m, "estimate"] - nd.loc[m, "ci_low"]], [nd.loc[m, "ci_high"] - nd.loc[m, "estimate"]]])
        for m in ("One-class WRMF", "One-class dual-regularized", "Popularity (most rated)"):
            if m in nd.index:
                ax.axhline(nd.loc[m, "estimate"], color=COLOR[m], ls="--", lw=1.3, label=f"{m} (no RMSE)")
        ax.invert_xaxis()
        ax.set_xlabel("RMSE (better → right)")
        ax.set_ylabel("NDCG@10 (better ↑)")
        ax.set_title(DISPLAY[ds])
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("The best rating predictor is not the best recommender", y=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    _save(fig, "fig_tradeoff.png")


# ----------------------------------------------------------------------- sensitivity
def load_val() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df = df[(df.split == "val") & (df.dataset.isin(DATASETS))].copy()
    p = pd.DataFrame(df.params.map(json.loads).tolist(), index=df.index)
    return pd.concat([df, p.add_prefix("p_")], axis=1)


def _best_by(df, col, metric, lower=False):
    g = df.groupby(col)[metric]
    return (g.min() if lower else g.max()).sort_index()


def fig_sensitivity_cf_mf(v: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
    for i, ds in enumerate(DATASETS):
        d = v[v.dataset == ds]
        for model, color, lab in (("item_knn", COLOR["Item-based CF"], "Item CF"), ("user_knn", COLOR["User-based CF"], "User CF")):
            c = d[(d.model == model) & (d.p_shrink == 25.0) & (d.notes == "tuning")]
            axes[i, 0].plot(*zip(*_best_by(c, "p_k", "rmse", True).items()), "o-", color=color, label=lab)
            axes[i, 1].plot(*zip(*_best_by(c, "p_k", "ndcg@10").items()), "o-", color=color, label=lab)
        mf = d[d.model == "mf_biased"]
        axes[i, 2].plot(*zip(*_best_by(mf, "p_k", "rmse", True).items()), "o-", color=COLOR["MF (biased)"], label="biased MF")
        mp = d[d.model == "mf_plain"]
        axes[i, 2].plot(*zip(*_best_by(mp, "p_k", "rmse", True).items()), "o--", color=COLOR["MF (plain)"], label="plain MF")
        ax, ax2 = axes[i, 3], axes[i, 3].twinx()
        ax.plot(*zip(*_best_by(mf, "p_lam", "rmse", True).items()), "o-", color=COLOR["MF (biased)"], label="best val RMSE")
        ax2.plot(*zip(*_best_by(mf, "p_lam", "ndcg@10").items()), "s--", color="#444444", label="best val NDCG@10")
        ax.set_ylabel("validation RMSE", color=COLOR["MF (biased)"])
        ax2.set_ylabel("validation NDCG@10")
        ax.set_xscale("log")
        titles = ["CF: validation RMSE vs neighbours k", "CF: validation NDCG@10 vs neighbours k",
                  "MF: validation RMSE vs latent dimension k", "Biased MF: RMSE and NDCG@10 vs λ"]
        for j in range(4):
            a = axes[i, j]
            a.set_title(f"{DISPLAY[ds]}\n{titles[j]}", fontsize=9)
            a.grid(alpha=0.3)
            a.set_xlabel("λ" if j == 3 else ("k" if j else "k (neighbours)"))
            if j < 3:
                a.set_xscale("log")
                a.legend(fontsize=8)
        axes[i, 0].set_ylabel("validation RMSE")
        axes[i, 1].set_ylabel("validation NDCG@10")
        axes[i, 2].set_ylabel("validation RMSE")
    fig.suptitle("Hyperparameter sensitivity (validation split; best value over the other hyperparameters; "
                 "user CF tuned on a 5,000-user sample)", y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_sensitivity_cf_mf.png")


def fig_sensitivity_oneclass(v: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
    for i, ds in enumerate(DATASETS):
        w = v[(v.dataset == ds) & (v.model == "occf_wrmf")]
        dual = v[(v.dataset == ds) & (v.model == "occf_dual")]
        for j, (col, label) in enumerate((("p_k", "latent dimension k"), ("p_alpha", "confidence α"),
                                         ("p_lam", "regularization λ"))):
            ax = axes[i, j]
            ser = _best_by(w, col, "ndcg@10")
            ax.plot(*zip(*ser.items()), "o-", color=COLOR["One-class WRMF"])
            ax.set_xscale("log")
            ax.set_xticks(list(ser.index), [f"{x:g}" for x in ser.index])
            ax.minorticks_off()
            ax.set_xlabel(label)
            ax.set_ylabel("validation NDCG@10")
            ax.set_title(f"{DISPLAY[ds]}\nWRMF vs {label}", fontsize=9)
            ax.grid(alpha=0.3)
        ax = axes[i, 3]
        base = w["ndcg@10"].max()
        dd = dual[(dual.p_lam_u == dual.p_lam_i) & (dual.p_lam_u > 0)].assign(strength=lambda d: d.p_lam_u)   # both graphs, equal strength
        s = dd.groupby("strength")["ndcg@10"].max().sort_index()
        xs = [0.0] + list(s.index)
        ax.plot(range(len(xs)), [base] + list(s.values), "o-", color=COLOR["One-class dual-regularized"])
        ax.set_xticks(range(len(xs)), ["0 (WRMF)"] + [f"{x:g}" for x in s.index])
        ax.set_xlabel("graph-regularization strength λu = λi")
        ax.set_ylabel("validation NDCG@10")
        ax.set_title(f"{DISPLAY[ds]}\nDual regularization vs graph strength", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("One-class CF sensitivity (validation split; best value over the other hyperparameters)", y=1.01,
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_sensitivity_oneclass.png")


# -------------------------------------------------------------------- learning curves
def fig_learning_curves() -> bool:
    files = [TAB / f"learning_curves_{ds}.csv" for ds in DATASETS]
    if not all(f.exists() for f in files):
        print("learning curves not available yet (run python -m src.curves for both datasets)")
        return False
    df = pd.concat([pd.read_csv(f) for f in files])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, ds in enumerate(DATASETS):
        d = df[df.dataset == ds]
        a = d[d.model == "MF (biased)"]
        ax = axes[0, j]
        ax.plot(a.step, a.train_rmse, "-", color="#999999", label="train RMSE (first 2M ratings)")
        ax.plot(a.step, a.val_rmse, "-", color=COLOR["MF (biased)"], label="validation RMSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{DISPLAY[ds]}\nBiased MF (final RMSE configuration)", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        o = d[d.model == "One-class dual-regularized"]
        ax = axes[1, j]
        ax.plot(o.step, o.val_ndcg10, "o-", color=COLOR["One-class dual-regularized"])
        ax.set_xlabel("ALS iteration")
        ax.set_ylabel("validation NDCG@10")
        ax.set_title(f"{DISPLAY[ds]}\nDual-regularized one-class CF (final configuration)", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_learning_curves.png")
    return True


# ----------------------------------------------------------------------------- index
INDEX = [
    ("Answers", [
        ("../findings.md", "Written answers to the five research questions with evidence, caveats and limitations."),
    ]),
    ("Results", [
        ("fig_main_results.png", "RMSE and NDCG@10 of every method with 95% bootstrap intervals (seed 42, test split)."),
        ("fig_tradeoff.png", "RMSE against NDCG@10: the best rating predictor (MF) is not the best recommender."),
        ("../tables/main_results.md", "Main comparison table (K = 10); `main_results_full.md` adds K = 5 and 20."),
        ("../tables/bootstrap_ci.md", "Point estimates with 95% bootstrap intervals."),
        ("../tables/paired_comparisons.md", "Paired bootstrap differences for the 17 headline comparisons."),
        ("../tables/seed_summary.md", "Mean ± standard deviation over three seeds."),
    ]),
    ("Low-history behaviour (research question 5)", [
        ("history_buckets.png", "Metrics by the number of the user's training ratings (natural variation, ≥20 ratings)."),
        ("../tables/history_buckets.md", "Same as tables, including NDCG@10 relative to popularity."),
        ("cold_start.png", "Simulated cold start: each user reveals only N = 1…50 ratings (MF and one-class via fold-in)."),
        ("../tables/cold_start.md", "Cold-start tables: NDCG@10, Hit@10, RMSE and the popularity-fallback share."),
    ]),
    ("Why the methods differ", [
        ("list_diagnostics.png", "Coverage, popularity of recommended items and share of less-rated items per method."),
        ("../tables/list_diagnostics.md", "Full list diagnostics (Gini, novelty, genre diversity …)."),
        ("../examples.md", "Success and failure examples with titles (rule-based selection), ML-25M."),
        ("../examples_analysis.md", "Interpretation of the examples."),
    ]),
    ("Model fitting", [
        ("fig_sensitivity_cf_mf.png", "Validation sensitivity to neighbours k, MF latent dimension and λ."),
        ("fig_sensitivity_oneclass.png", "Validation sensitivity of one-class CF to k, α, λ and the dual-regularization strength (the strength-1 point comes from an earlier tuning round with a smaller base λ, so that curve is indicative)."),
        ("fig_learning_curves.png", "MF train/validation RMSE per epoch; one-class validation NDCG@10 per iteration."),
    ]),
    ("Data (EDA)", [
        ("eda_rating_distribution.png", "Rating distributions of the datasets."),
        ("eda_item_popularity.png", "Long-tailed item popularity."),
        ("eda_user_activity.png", "User activity distribution."),
        ("eda_history_buckets.png", "Number of users per history bucket."),
        ("eda_ml_genres.png", "MovieLens genre frequencies."),
        ("eda_time.png", "Ratings over time."),
        ("eda_kaggle_titles.png", "Kaggle Netflix titles metadata overview."),
        ("../eda_tables.md", "EDA summary tables (`eda_summary.json` has the numbers)."),
    ]),
]


def write_index() -> None:
    lines = ["# Figures and tables available for the report\n",
             "Paths are relative to `reports/figures/`. Everything is regenerated by the commands in the README.\n"]
    for section, items in INDEX:
        lines.append(f"## {section}\n")
        for path, cap in items:
            ok = (FIG / path).resolve().exists()
            lines.append(f"- [{path}]({path}) — {cap}" + ("" if ok else "  **(not generated yet)**"))
        lines.append("")
    (FIG / "INDEX.md").write_text("\n".join(lines))
    print("wrote INDEX.md")


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    order = list(cfg["datasets"]["ml_25m"].keys())
    ci = pd.read_csv(TAB / "bootstrap_ci.csv")
    fig_main_results(ci, order)
    fig_tradeoff(ci)
    v = load_val()
    fig_sensitivity_cf_mf(v)
    fig_sensitivity_oneclass(v)
    fig_learning_curves()
    write_index()


if __name__ == "__main__":
    main()
