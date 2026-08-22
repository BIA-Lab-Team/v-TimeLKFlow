"""Tests for the windowed VAR(1) regression-coefficient variance estimator
(Hagan et al. 2019, appendix C)."""

import numpy as np

from mtvlk.utils.regression_se import RegressionSE


def _reference_variance(X, dX, C):
    """From-scratch OLS fit + delta-method calculation, for comparison
    against RegressionSE's incremental running-sum implementation."""
    n = X.shape[1]
    design = np.hstack([np.ones((X.shape[0], 1)), X])
    XtX_inv = np.linalg.inv(design.T @ design)
    Var_dC = np.zeros((n, n))
    for i in range(n):
        y = dX[:, i]
        beta = XtX_inv @ design.T @ y
        resid = y - design @ beta
        dof = X.shape[0] - (n + 1)
        sigma2 = (resid @ resid) / dof
        Cov_a_i = sigma2 * XtX_inv[1:, 1:]
        for j in range(n):
            Var_dC[j, i] = C[j, :] @ Cov_a_i @ C[j, :]
    return Var_dC


class TestRegressionSE:
    def test_matches_from_scratch_ols(self):
        rng = np.random.default_rng(0)
        n = 3
        window = 200
        X = rng.standard_normal((window, n))
        dX = rng.standard_normal((window, n)) * 0.1
        C = np.cov(X.T, ddof=1)

        est = RegressionSE(n, window)
        est.warm_start(X, dX)
        Var_dC = est.variance(C)

        expected = _reference_variance(X, dX, C)
        np.testing.assert_allclose(Var_dC, expected, rtol=1e-8, atol=1e-12)

    def test_returns_none_with_too_few_samples(self):
        n = 3
        est = RegressionSE(n, window=200)
        est.update(np.zeros(n), np.zeros(n))
        assert est.variance(np.eye(n)) is None

    def test_positive_and_finite_after_warmup(self):
        rng = np.random.default_rng(1)
        n = 2
        window = 100
        est = RegressionSE(n, window)
        C = np.eye(n)
        for _ in range(window):
            x = rng.standard_normal(n)
            dx = rng.standard_normal(n) * 0.1
            est.update(x, dx)
        Var_dC = est.variance(C)
        assert Var_dC is not None
        assert Var_dC.shape == (n, n)
        assert np.all(np.isfinite(Var_dC))
        assert np.all(Var_dC >= 0)

    def test_sliding_window_matches_batch_refit(self):
        """After the ring buffer is full and slides, the running sums should
        match a from-scratch fit over the *current* window only (not the
        full history)."""
        rng = np.random.default_rng(2)
        n = 2
        window = 50
        est = RegressionSE(n, window)

        all_X, all_dX = [], []
        for _ in range(window + 30):
            x = rng.standard_normal(n)
            dx = rng.standard_normal(n) * 0.1
            est.update(x, dx)
            all_X.append(x)
            all_dX.append(dx)

        X_win = np.array(all_X[-window:])
        dX_win = np.array(all_dX[-window:])
        C = np.cov(X_win.T, ddof=1)

        Var_dC = est.variance(C)
        expected = _reference_variance(X_win, dX_win, C)
        np.testing.assert_allclose(Var_dC, expected, rtol=1e-8, atol=1e-12)
