"""Tests for the static multivariate LK (Liang 2021)."""

import numpy as np
import pytest

from mtvlk.core.lk_stationary import compute_lk, _lk_from_moments
from mtvlk.utils.covariance import (
    finite_difference,
    sample_covariance,
    tendency_covariance,
    cofactor_matrix,
    pack_upper,
    unpack_upper,
)


def make_ar1_coupled(N: int, a11: float, a12: float, rng) -> np.ndarray:
    """Two-variable AR(1): x1[t+1] = a11*x1[t] + noise, x2[t+1] = a12*x1[t] + noise.

    Causal: x1 -> x2 (a12 != 0), x2 does NOT cause x1.
    """
    X = np.zeros((N, 2))
    sigma = 0.5
    for t in range(N - 1):
        X[t + 1, 0] = a11 * X[t, 0] + sigma * rng.standard_normal()
        X[t + 1, 1] = a12 * X[t, 0] + sigma * rng.standard_normal()
    return X


class TestCovarianceHelpers:
    def test_finite_difference_shape(self):
        X = np.ones((10, 3))
        dX = finite_difference(X, dt=0.1)
        assert dX.shape == (9, 3)

    def test_finite_difference_constant(self):
        X = np.tile(np.arange(5), (3, 1)).T.astype(float)  # linear ramp
        dX = finite_difference(X, dt=1.0)
        np.testing.assert_allclose(dX, np.ones((4, 3)))

    def test_sample_covariance_symmetric(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        C = sample_covariance(X)
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        assert C.shape == (4, 4)

    def test_cofactor_matrix_identity(self):
        C = np.eye(3)
        Delta = cofactor_matrix(C)
        # For identity: det=1, inv=I, so Delta = I.T = I
        np.testing.assert_allclose(Delta, np.eye(3), atol=1e-12)

    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(7)
        M = rng.standard_normal((4, 4))
        M = M + M.T  # symmetrise
        v = pack_upper(M)
        M2 = unpack_upper(v, 4)
        np.testing.assert_allclose(M, M2, atol=1e-12)


class TestLKStationary:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 3))
        res = compute_lk(X, dt=1.0, n_boot=0)
        assert res["T"].shape == (3, 3)
        assert res["tau"].shape == (3, 3)

    def test_tau_bounded(self):
        """tau values should generally be in [-100, 100]%."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 3))
        res = compute_lk(X, dt=1.0, n_boot=0)
        assert np.all(np.abs(res["tau"]) <= 100 + 1e-6)

    def test_direction_detection(self):
        """x1 causes x2 (a12=0.5), x2 does not cause x1 (a21=0).
        Should detect T[0,1] > T[1,0] (flow from 0 to 1 > flow from 1 to 0).
        T[j,i] = T_{j->i}: T[0,1] means j=0 (x1) -> i=1 (x2).
        """
        rng = np.random.default_rng(2)
        X = make_ar1_coupled(5000, a11=0.5, a12=0.5, rng=rng)
        res = compute_lk(X, dt=1.0, n_boot=0)
        T = res["T"]
        # T[0,1] = T_{x1->x2} should dominate T[1,0] = T_{x2->x1}
        assert T[0, 1] > T[1, 0], f"Expected T[0,1]={T[0,1]:.4f} > T[1,0]={T[1,0]:.4f}"

    def test_no_causality(self):
        """Independent variables should have near-zero off-diagonal flow."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((2000, 3))
        res = compute_lk(X, dt=1.0, n_boot=0)
        T = res["T"]
        # Off-diagonal should be small relative to diagonal
        diag_scale = np.mean(np.abs(np.diag(T)))
        off_diag = T.copy()
        np.fill_diagonal(off_diag, 0)
        assert np.max(np.abs(off_diag)) < 2 * diag_scale

    def test_bootstrap_errors_shape(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((100, 2))
        res = compute_lk(X, dt=1.0, n_boot=20, rng=rng)
        assert res["T_err"].shape == (2, 2)
        assert res["tau_err"].shape == (2, 2)
        assert np.all(res["T_err"] >= 0)

    def test_lk_from_moments_singular_raises(self):
        C = np.zeros((3, 3))  # singular
        dC = np.eye(3)
        with pytest.raises(np.linalg.LinAlgError):
            _lk_from_moments(C, dC)
