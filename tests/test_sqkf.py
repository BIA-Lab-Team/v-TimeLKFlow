"""Tests for the Square-Root Kalman Filter."""

import numpy as np
import pytest

from mtvlk.core.sqkf import SquareRootKF
from mtvlk.utils.covariance import finite_difference


class TestSquareRootKF:
    def test_state_dimension(self):
        n = 3
        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        expected_m = n * (n + 1) // 2 + n * n
        assert kf.m == expected_m
        assert kf.theta.shape == (expected_m,)
        assert kf.S.shape == (expected_m, expected_m)

    def test_initialize_sets_state(self):
        rng = np.random.default_rng(42)
        n = 3
        T0 = 50
        X_init = rng.standard_normal((T0, n))
        dX_init = np.diff(X_init, axis=0)
        dX_init = np.vstack([dX_init, dX_init[-1:]])  # pad to same length

        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        kf.initialize(X_init, dX_init)

        # State should be non-trivial after initialization
        assert not np.all(kf.theta == 0)
        # S should be full rank
        assert np.linalg.matrix_rank(kf.S) == kf.m

    def test_predict_increases_uncertainty(self):
        rng = np.random.default_rng(0)
        n = 2
        kf = SquareRootKF(n, Q_scale=0.01, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        kf.initialize(X_init[:29], dX_init)

        P_before = kf.S @ kf.S.T
        kf.predict()
        P_after = kf.S @ kf.S.T

        # Trace of P should increase after predict (uncertainty grows)
        assert np.trace(P_after) > np.trace(P_before)

    def test_update_reduces_uncertainty(self):
        rng = np.random.default_rng(1)
        n = 2
        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        kf.initialize(X_init[:29], dX_init)
        kf.predict()

        P_before = kf.S @ kf.S.T
        x = rng.standard_normal(n)
        dx = rng.standard_normal(n)
        mu_x = X_init.mean(axis=0)
        mu_dx = dX_init.mean(axis=0)
        y = kf.build_observation(x, dx, mu_x, mu_dx)
        kf.update(y)
        P_after = kf.S @ kf.S.T

        assert np.trace(P_after) < np.trace(P_before)

    def test_extract_moments_symmetric_C(self):
        rng = np.random.default_rng(5)
        n = 3
        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        X_init = rng.standard_normal((40, n))
        dX_init = finite_difference(X_init, 1.0)
        kf.initialize(X_init[:39], dX_init)
        C, dC = kf.extract_moments()

        assert C.shape == (n, n)
        assert dC.shape == (n, n)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_stationary_convergence(self):
        """On a long stationary series, KF estimate of C should converge to sample C."""
        rng = np.random.default_rng(99)
        n = 2
        N = 2000
        X = rng.standard_normal((N, n))
        dX = finite_difference(X, 1.0)

        kf = SquareRootKF(n, Q_scale=1e-6, R_scale=1.0)
        kf.initialize(X[:30], dX[:30])

        mu_x = X[:30].mean(axis=0)
        mu_dx = dX[:30].mean(axis=0)

        for t in range(30, N - 1):
            kf.predict()
            y = kf.build_observation(X[t], dX[t], mu_x, mu_dx)
            kf.update(y)
            mu_x = mu_x + (X[t] - mu_x) / (t + 1)
            mu_dx = mu_dx + (dX[t] - mu_dx) / (t + 1)

        C_kf, _ = kf.extract_moments()

        # Sample covariance of the full series
        C_sample = np.cov(X.T, ddof=1)

        # Should be within ~20% of sample estimate
        np.testing.assert_allclose(C_kf, C_sample, rtol=0.25, atol=0.1)

    def test_build_observation_shape(self):
        n = 4
        kf = SquareRootKF(n)
        x = np.ones(n)
        dx = np.ones(n)
        mu_x = np.zeros(n)
        mu_dx = np.zeros(n)
        y = kf.build_observation(x, dx, mu_x, mu_dx)
        assert y.shape == (kf.m,)

    def test_predict_with_explicit_Q_overrides_default(self):
        rng = np.random.default_rng(2)
        n = 2
        kf_default = SquareRootKF(n, Q_scale=1e-6, R_scale=1.0)
        kf_explicit = SquareRootKF(n, Q_scale=1e-6, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        for kf in (kf_default, kf_explicit):
            kf.initialize(X_init[:29], dX_init)

        P_before = kf_default.S.T @ kf_default.S

        kf_default.predict()
        big_Q = np.full(kf_explicit.m, 10.0)
        kf_explicit.predict(big_Q)

        P_default = kf_default.S.T @ kf_default.S
        P_explicit = kf_explicit.S.T @ kf_explicit.S

        # Explicit large Q should grow uncertainty far more than the tiny
        # default Q_scale.
        assert np.trace(P_explicit) > np.trace(P_default) > np.trace(P_before)

    def test_predict_no_args_unchanged(self):
        """Calling predict() with no args must behave exactly as before."""
        rng = np.random.default_rng(3)
        n = 2
        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        kf.initialize(X_init[:29], dX_init)

        P_before = kf.S.T @ kf.S
        kf.predict()
        P_after = kf.S.T @ kf.S
        np.testing.assert_allclose(P_after, P_before + kf._Q)

    def test_update_with_explicit_R_smaller_pulls_closer_to_observation(self):
        rng = np.random.default_rng(4)
        n = 2
        kf_small_R = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        kf_large_R = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        for kf in (kf_small_R, kf_large_R):
            kf.initialize(X_init[:29], dX_init)
            kf.predict()

        x = rng.standard_normal(n)
        dx = rng.standard_normal(n)
        mu_x = X_init.mean(axis=0)
        mu_dx = dX_init.mean(axis=0)
        y = kf_small_R.build_observation(x, dx, mu_x, mu_dx)

        theta_prior = kf_small_R.theta.copy()

        kf_small_R.update(y, np.full(kf_small_R.m, 1e-6))
        kf_large_R.update(y, np.full(kf_large_R.m, 1e6))

        dist_small = np.linalg.norm(kf_small_R.theta - y)
        dist_large = np.linalg.norm(kf_large_R.theta - y)
        assert dist_small < dist_large

    def test_update_no_extra_arg_unchanged(self):
        """Calling update(y) with no R override must behave exactly as before."""
        rng = np.random.default_rng(5)
        n = 2
        kf = SquareRootKF(n, Q_scale=1e-4, R_scale=1.0)
        X_init = rng.standard_normal((30, n))
        dX_init = finite_difference(X_init, 1.0)
        kf.initialize(X_init[:29], dX_init)
        kf.predict()

        x = rng.standard_normal(n)
        dx = rng.standard_normal(n)
        mu_x = X_init.mean(axis=0)
        mu_dx = dX_init.mean(axis=0)
        y = kf.build_observation(x, dx, mu_x, mu_dx)

        P_before = kf.S.T @ kf.S
        theta_before = kf.theta.copy()
        kf.update(y)

        # Recompute expected result by hand using the standard KF equations.
        nu = y - theta_before
        P_e = P_before + kf._R
        K = np.linalg.solve(P_e.T, P_before.T).T
        expected_theta = theta_before + K @ nu
        np.testing.assert_allclose(kf.theta, expected_theta, atol=1e-10)

    def test_analytical_R_matches_hand_formula(self):
        n = 3
        kf = SquareRootKF(n)
        rng = np.random.default_rng(6)
        A = rng.standard_normal((n, n))
        C = A @ A.T + np.eye(n)  # random SPD covariance
        dC = rng.standard_normal((n, n)) * 0.1
        D_diag = np.array([1.0, 2.0, 0.5])

        R = kf.analytical_R(C, dC, D_diag)
        assert R.shape == (kf.m,)

        k_idx, i_idx = kf._triu_idx
        for idx, (k, i) in enumerate(zip(k_idx, i_idx)):
            if k == i:
                expected = 2 * C[k, k] ** 2
            else:
                expected = C[k, k] * C[i, i] + C[k, i] ** 2
            assert R[idx] == pytest.approx(expected)

        dC_offset = kf.n_C
        for k in range(n):
            for i in range(n):
                idx = dC_offset + k * n + i
                expected = C[k, k] * D_diag[i] + dC[k, i] ** 2
                assert R[idx] == pytest.approx(expected)

    def test_analytical_R_floored_and_positive(self):
        n = 2
        kf = SquareRootKF(n)
        C = np.zeros((n, n))
        dC = np.zeros((n, n))
        D_diag = np.zeros(n)
        R = kf.analytical_R(C, dC, D_diag, R_floor=1e-8)
        assert np.all(R >= 1e-8)
