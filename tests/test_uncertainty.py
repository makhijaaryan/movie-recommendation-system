import numpy as np

from src.uncertainty import boot_sums, ci


def test_bootstrap_mean_ci_covers_truth_and_scales_with_n():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, size=4000)
    X = np.column_stack([x, np.ones_like(x)])
    s = boot_sums(X, B=400, seed=3)
    means = s[:, 0] / s[:, 1]
    lo, hi = ci(means)
    assert lo < x.mean() < hi
    assert np.isclose(s[:, 1].mean(), len(x))                  # each resample has n draws
    assert abs((hi - lo) - 2 * 1.96 / np.sqrt(len(x))) < 0.02  # ≈ analytic width of the CI of a mean


def test_paired_difference_of_identical_methods_is_exactly_zero():
    x = np.random.default_rng(1).random(500)
    X = np.column_stack([x, x, np.ones_like(x)])               # "method A" and "method B" identical
    s = boot_sums(X, B=100, seed=5)
    d = s[:, 0] / s[:, 2] - s[:, 1] / s[:, 2]
    assert np.all(d == 0)


def test_pairing_detects_a_small_consistent_improvement():
    rng = np.random.default_rng(2)
    base = rng.random(3000)
    a, b = base + 0.01, base                                   # A always 0.01 better, but per-user noise is large
    X = np.column_stack([a, b, np.ones_like(a)])
    s = boot_sums(X, B=300, seed=7)
    lo, hi = ci(s[:, 0] / s[:, 2] - s[:, 1] / s[:, 2])
    assert lo > 0 and abs((lo + hi) / 2 - 0.01) < 1e-3
