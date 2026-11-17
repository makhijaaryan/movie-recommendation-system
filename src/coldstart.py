"""Simulated cold start (task 6.4, research question 5).

Every evaluated user keeps only N randomly chosen TRAIN ratings (nested: the N=3 ratings are a subset of
the N=5 ratings, ...), N in {1, 3, 5, 10, 20, 50}; users with fewer than N train ratings keep all of them.
Everything else is identical to the main protocol: same held-out test ratings, same relevance rule, same
excluded (already seen) items, same frozen models and hyperparameters. What changes is only what the
recommender is allowed to know about the user:

  * popularity / mean rating: independent of the user -> evaluated once (flat reference lines)
  * item CF, user CF: the target user's row is replaced by the truncated history; item similarities and the
    neighbour pool keep their full-history training data (as for a new user joining a live system)
  * MF, one-class CF: the user's vector is folded in from the N ratings with the item factors fixed
    (src/foldin.py)
Rows where a model has no usable signal (all scores equal, e.g. one-class CF with no rating >= 4 among the
N kept, or CF with no overlapping neighbours) fall back to the popularity ranking; the share is reported.
Rating prediction (RMSE/MAE) uses the model's own prediction (CF: user mean + neighbour deviation).

Users: for MovieLens 25M a fixed random subset of eligible users (--users, default 30,000) is used for the
whole sweep (user-based CF over all 152k users costs ~50 min per N); Netflix uses all eligible users. The
same users are used for every model and N, including the reference point N = all (full history).

Outputs: reports/tables/cold_start_<dataset>.csv (and a log row per model/N in experiments/results.csv with
split 'test_cold'); `python -m src.coldstart --report` builds cold_start.md and the figure from the CSVs.
Run:  python -m src.coldstart --dataset netflix
"""
from __future__ import annotations

import argparse
import copy
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
from src.eval import RESULTS_CSV, evaluate_topk, log_result
from src.final_run import fit_model, fit_key, job_key, logged_params, per_user_rating
from src.final_tables import DISPLAY
from src.foldin import foldin_mf, foldin_occf
from src.uncertainty import boot_sums, ci

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "final_config.json"
OUT = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
NS = (1, 3, 5, 10, 20, 50)          # D29
ALL = "all"
INDEPENDENT = ("global_mean", "mean_rating", "popularity_count")      # do not use the user's history
B = 300


# ----------------------------------------------------------------------------- helpers
def truncate_histories(train: pd.DataFrame, n_users: int, n_items: int, n_keep: int, seed: int = SEED) -> sp.csr_matrix:
    """CSR of at most `n_keep` randomly chosen train ratings per user (nested across n_keep for a fixed seed)."""
    key = np.random.default_rng(seed + 1000).random(len(train))
    u = train["u"].to_numpy()
    order = np.lexsort((key, u))
    us = u[order]
    counts = np.bincount(us, minlength=n_users)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = np.arange(len(train)) - starts[us]
    rows = order[pos < n_keep]
    sub = train.iloc[rows]
    return sp.csr_matrix((sub["rating"].to_numpy(np.float32), (sub["u"].to_numpy(), sub["i"].to_numpy())),
                         shape=(n_users, n_items))


class Fallback:
    """score_fn wrapper: users whose score row carries no information get the popularity ranking."""

    def __init__(self, fn, pop):
        self.fn, self.pop, self.bad, self.n = fn, pop, 0, 0

    def __call__(self, users):
        s = np.array(self.fn(users), dtype=np.float32)
        bad = s.max(axis=1) <= s.min(axis=1)
        s[bad] = self.pop
        self.bad += int(bad.sum())
        self.n += len(users)
        return s

    @property
    def frac(self):
        return self.bad / max(self.n, 1)


def make_variant(model, spec, N, ctx, trunc):
    """The model as it would score users who only revealed `N` ratings."""
    m = spec["model"]
    if N == ALL or m in INDEPENDENT:
        return model
    users = ctx["users"]
    if m in ("item_knn", "user_knn"):
        return model.with_targets(trunc)
    c = copy.copy(model)
    P_full = np.zeros((ctx["n_users"], model.P.shape[1]), dtype=np.float32)
    if m.startswith("mf"):
        P, bu = foldin_mf(model, trunc, users)
        P_full[users] = P
        bu_full = np.zeros(ctx["n_users"], dtype=np.float32)
        bu_full[users] = bu
        c.bu = bu_full
    else:
        P_full[users] = foldin_occf(model, trunc, users)
    c.P = P_full
    return c


def set_k(variant, spec, role):
    if spec["model"] not in ("item_knn", "user_knn"):
        return
    kk = spec["params"]["k_rating" if role == "rating" else "k_rank"]
    if spec["model"] == "item_knn":
        if variant.k != kk:
            variant.set_k(kk)
    elif variant.k != kk:
        variant.k = kk
        variant.sk_cache = None


def evaluate(variant, spec, roles, ctx):
    """Metrics of one model variant for the given roles + per-user arrays for the bootstrap."""
    out: dict = {}
    if "rating" in roles:
        set_k(variant, spec, "rating")
        pu = per_user_rating(variant, ctx["test"], ctx["lo"], ctx["hi"])
        s = boot_sums(pu[["sse", "sae", "n"]].to_numpy(float), B, seed=11)
        out["rmse"] = float(np.sqrt(pu.sse.sum() / pu.n.sum()))
        out["mae"] = float(pu.sae.sum() / pu.n.sum())
        lo, hi = ci(np.sqrt(s[:, 0] / s[:, 2]))
        out["rmse_lo"], out["rmse_hi"], out["n_users_rating"] = lo, hi, len(pu)
    if "ranking" in roles:
        set_k(variant, spec, "ranking")
        fb = Fallback(variant.score_fn, ctx["pop"])
        res, pk = evaluate_topk(fb, ctx["test"], ctx["trval"], ctx["n_items"], return_per_user=True)
        out.update({k: res[k] for k in ("precision@10", "recall@10", "hit@10", "ndcg@10")})
        for m in ("ndcg@10", "hit@10"):
            X = np.column_stack([pk[m].to_numpy(), np.ones(len(pk))])
            s = boot_sums(X, B, seed=12)
            out[m.split("@")[0] + "_lo"], out[m.split("@")[0] + "_hi"] = ci(s[:, 0] / s[:, 1])
        out["n_users_rank"], out["frac_fallback"] = res["n_users"], fb.frac
    return out


# ------------------------------------------------------------------------------- driver
def run(dataset: str, n_users_cap: int | None, ns, config_path: str, only=None, results=RESULTS_CSV) -> pd.DataFrame:
    cfg = json.loads(Path(config_path).read_text())
    methods = cfg["datasets"][dataset]
    lo, hi = (0.5, 5.0) if dataset.startswith("ml") else (1.0, 5.0)
    train, val, test_full = (data.load_split(dataset, s) for s in ("train", "val", "test"))
    tr = data.load_train_csr(dataset)
    n_users, n_items = tr.shape
    trval = tr + sp.csr_matrix((np.ones(len(val), dtype=np.float32), (val["u"], val["i"])), shape=tr.shape)

    eligible = np.sort(test_full.loc[test_full["rating"] >= 4.0, "u"].unique())
    users = eligible
    if n_users_cap and len(eligible) > n_users_cap:
        users = np.sort(np.random.default_rng(SEED).choice(eligible, n_users_cap, replace=False))
    test = test_full[test_full["u"].isin(users)].reset_index(drop=True)
    pop = np.bincount(train["i"], minlength=n_items).astype(np.float32)
    ctx = dict(dataset=dataset, train=train, val=val, n_users=n_users, n_items=n_items, tr=tr, trval=trval,
               test=test, users=users, lo=lo, hi=hi, pop=pop)
    print(f"{dataset}: {len(users):,} users, {len(test):,} test ratings, N = {list(ns)}", flush=True)

    truncs = {N: truncate_histories(train, n_users, n_items, N) for N in ns}
    kept = {N: float(np.minimum(np.diff(tr.indptr)[users], N).mean()) for N in ns}
    print("mean ratings kept per user:", {N: round(v, 1) for N, v in kept.items()}, flush=True)

    jobs: dict[str, dict] = {}
    for label, spec in methods.items():
        if only and label not in only:
            continue
        for role in ("rating", "ranking"):
            if role in spec:
                j = jobs.setdefault(job_key(spec[role]), dict(spec=spec[role], uses=[], roles=set()))
                j["uses"].append((label, role))
                j["roles"].add(role)

    rows: dict[tuple, dict] = {}                      # (label, N) -> merged metrics
    model, model_key = None, None
    for j in sorted(jobs.values(), key=lambda j: fit_key(j["spec"])):
        spec, t0 = j["spec"], time.time()
        if fit_key(spec) != model_key:
            model = None
            model = fit_model(spec, ctx)
            model_key = fit_key(spec)
            print(f"fitted {model_key[:60]} in {time.time()-t0:.0f}s", flush=True)
        grid = [ALL] if spec["model"] in INDEPENDENT else [*ns, ALL]
        for N in grid:
            t1 = time.time()
            variant = make_variant(model, spec, N, ctx, truncs.get(N))
            # rating/ranking roles are separate configs only for some methods; evaluate what this job serves
            m = evaluate(variant, spec, j["roles"], ctx)
            for label, role in j["uses"]:
                rows.setdefault((label, N), dict(dataset=dataset, method=label, N=N, users=len(users),
                                                 mean_kept=kept.get(N, np.nan)))
                keys = ("rmse", "mae", "rmse_lo", "rmse_hi", "n_users_rating") if role == "rating" else \
                       ("precision@10", "recall@10", "hit@10", "ndcg@10", "ndcg_lo", "ndcg_hi", "hit_lo", "hit_hi",
                        "n_users_rank", "frac_fallback")
                rows[(label, N)].update({k: m[k] for k in keys if k in m})
            log_result(dataset, spec["model"], "test_cold", {**logged_params(spec), "cold_n": N,
                                                              "eval_users": len(users)},
                       {k: v for k, v in m.items() if not k.endswith(("_lo", "_hi"))}, notes="cold-start", path=results)
            print(f"  {spec['model']:<14} N={str(N):<4} " + " ".join(
                f"{k}={m[k]:.4f}" for k in ("rmse", "ndcg@10", "hit@10") if k in m)
                  + (f" fallback={m['frac_fallback']:.2f}" if "frac_fallback" in m else "")
                  + f" [{time.time()-t1:.0f}s]", flush=True)
    df = pd.DataFrame(rows.values())
    df["N"] = df["N"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"cold_start_{dataset}.csv", index=False)
    return df


# ------------------------------------------------------------------------------- report
STYLE = {"Popularity (most rated)": ("#888888", "--"), "Mean rating (min 50 ratings)": ("#bbbbbb", ":"),
         "Global mean": ("#cccccc", ":"), "Item-based CF": ("#1f77b4", "-"), "User-based CF": ("#17becf", "-"),
         "MF (biased)": ("#d62728", "-"), "MF (plain)": ("#ff9896", "-"),
         "One-class WRMF": ("#2ca02c", "-"), "One-class dual-regularized": ("#006400", "-")}


def report() -> None:
    frames = [pd.read_csv(f) for f in sorted(OUT.glob("cold_start_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["N"] = df["N"].astype(str)
    order = [str(n) for n in NS] + [ALL]
    md = ["Simulated cold start: each user reveals only N random training ratings (nested), models and "
          "hyperparameters frozen; 'all' = full history on the same users. Test relevance and excluded items "
          "are unchanged. Fallback = share of users with no usable signal who got the popularity ranking.\n"]
    for metric, title, fmt in (("ndcg@10", "NDCG@10", "{:.4f}"), ("hit@10", "Hit@10", "{:.4f}"),
                               ("rmse", "RMSE", "{:.4f}"), ("frac_fallback", "Fallback share (ranking)", "{:.2f}")):
        for ds, g in df.groupby("dataset", sort=False):
            if metric not in g or g[metric].isna().all():
                continue
            n_users = int(g["users"].iloc[0])
            md.append(f"**{DISPLAY.get(ds, ds)} — {title}** ({n_users:,} users)\n")
            md.append("| Method | " + " | ".join(order) + " |")
            md.append("|:--|" + "---:|" * len(order))
            for label in dict.fromkeys(g.method):
                r = g[g.method == label].set_index("N")
                if r[metric].isna().all():
                    continue
                vals = []
                for n in order:
                    if n in r.index and pd.notna(r.loc[n, metric]):
                        vals.append(fmt.format(r.loc[n, metric]))
                    elif r.index.tolist() == [ALL] and pd.notna(r.loc[ALL, metric]):
                        vals.append(fmt.format(r.loc[ALL, metric]))      # user-independent: same for every N
                    else:
                        vals.append("–")
                md.append(f"| {label} | " + " | ".join(vals) + " |")
            md.append("")
    (OUT / "cold_start.md").write_text("\n".join(md))

    FIG.mkdir(parents=True, exist_ok=True)
    datasets = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.6 * len(datasets), 7.6), squeeze=False)
    x = {n: i for i, n in enumerate(order)}
    for j, ds in enumerate(datasets):
        for i, (metric, ylabel) in enumerate((("ndcg@10", "NDCG@10 (higher is better)"),
                                              ("rmse", "RMSE (lower is better)"))):
            ax = axes[i, j]
            g = df[df.dataset == ds]
            for label in dict.fromkeys(g.method):
                r = g[g.method == label].set_index("N")
                if r[metric].isna().all():
                    continue
                c, ls = STYLE.get(label, ("k", "-"))
                if r.index.tolist() == [ALL]:
                    ax.axhline(r.loc[ALL, metric], color=c, ls=ls, lw=1.4, label=label)
                    continue
                pts = [n for n in order if n in r.index and pd.notna(r.loc[n, metric])]
                ax.plot([x[n] for n in pts], [r.loc[n, metric] for n in pts], ls, color=c, marker="o", ms=4,
                        label=label)
                lo, hi = ("ndcg_lo", "ndcg_hi") if metric == "ndcg@10" else ("rmse_lo", "rmse_hi")
                ax.fill_between([x[n] for n in pts], [r.loc[n, lo] for n in pts], [r.loc[n, hi] for n in pts],
                                color=c, alpha=0.15, lw=0)
            ax.set_xticks(range(len(order)), ["full" if n == ALL else n for n in order])
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(DISPLAY.get(ds, ds))
            else:
                ax.set_xlabel("number of ratings the user has revealed (N)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "cold_start.png", dpi=160)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--users", type=int, default=30000, help="cap on evaluated users (0 = all)")
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--results", default=str(RESULTS_CSV))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a.dataset, a.users or None, a.ns, a.config, a.only, Path(a.results))
