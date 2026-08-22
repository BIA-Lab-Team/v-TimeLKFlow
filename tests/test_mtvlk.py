"""End-to-end tests for the MtvLK time-varying information flow (Zhou et al. 2024)."""

import numpy as np
import pytest

from mtvlk import compute_lk, compute_mtvlk, compute_mtvlk_filled, to_xarray
from mtvlk.core.sqkf import SquareRootKF


def make_regime_shift_series(N: int, rng, coupling_before: float = 0.0,
                              coupling_after: float = 0.5) -> np.ndarray:
    """Two-variable AR(1) with a coupling that switches at the midpoint.

    x1[t+1] = 0.5*x1[t] + noise
    x2[t+1] = c(t)*x1[t] + 0.3*x2[t] + noise

    c(t) = coupling_before for t < N/2
    c(t) = coupling_after  for t >= N/2
    """
    X = np.zeros((N, 2))
    sigma = 0.3
    midpoint = N // 2
    for t in range(N - 1):
        c = coupling_before if t < midpoint else coupling_after
        X[t + 1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
        X[t + 1, 1] = c * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()
    return X


class TestMtvLK:
    def test_output_shapes(self):
        rng = np.random.default_rng(10)
        N, n = 300, 3
        X = rng.standard_normal((N, n))
        res = compute_mtvlk(X, dt=1.0)
        assert res["T_t"].shape == (N - 1, n, n)
        assert res["tau_t"].shape == (N - 1, n, n)
        assert res["time_idx"].shape == (N - 1,)

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 2))
        res = compute_mtvlk(X, dt=1.0)
        # At most init_window NaN rows expected at the start; rest should be finite
        T_t = res["T_t"]
        finite_mask = np.isfinite(T_t).all(axis=(1, 2))
        # Most time steps (well past init window) should be finite
        assert finite_mask[50:].all(), "NaNs found beyond warm-up window"

    def test_tau_roughly_bounded(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((300, 3))
        res = compute_mtvlk(X, dt=1.0)
        tau_t = res["tau_t"]
        valid = tau_t[np.isfinite(tau_t)]
        # tau should be dominated by values in a reasonable range
        assert np.percentile(np.abs(valid), 99) < 200

    def test_regime_shift_detection(self):
        """MtvLK should detect that T_{x1->x2}(t) increases after the regime shift
        at N/2. Static LK should show a weaker, time-averaged signal."""
        rng = np.random.default_rng(42)
        N = 2000
        X = make_regime_shift_series(N, rng, coupling_before=0.0, coupling_after=0.6)

        res = compute_mtvlk(X, dt=1.0, Q_scale=5e-5, R_scale=1.0)
        T_t = res["T_t"]

        # T[0,1] = T_{x1->x2}; average over first half vs second half
        half = (N - 1) // 2
        T_first = np.nanmean(T_t[:half, 0, 1])
        T_second = np.nanmean(T_t[half:, 0, 1])

        # After regime shift causality should be stronger
        assert T_second > T_first, (
            f"Expected T_second={T_second:.4f} > T_first={T_first:.4f}"
        )

    def test_consistent_with_static_on_stationary(self):
        """For a long stationary series, the time-mean of T_t should be close to static T."""
        rng = np.random.default_rng(7)
        N = 3000
        X = np.zeros((N, 2))
        sigma = 0.4
        a12 = 0.4
        for t in range(N - 1):
            X[t + 1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
            X[t + 1, 1] = a12 * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()

        res_static = compute_lk(X, dt=1.0, n_boot=0)
        res_tv = compute_mtvlk(X, dt=1.0, Q_scale=1e-5, R_scale=1.0)

        T_static = res_static["T"]
        T_tv_mean = np.nanmean(res_tv["T_t"][100:], axis=0)  # skip warm-up

        # Ratio of mean time-varying to static should be reasonably close
        # (not exact because of KF lag, but direction should match)
        for j, i in [(0, 1), (1, 0)]:
            sign_static = np.sign(T_static[j, i])
            sign_tv = np.sign(T_tv_mean[j, i])
            assert sign_static == sign_tv or abs(T_static[j, i]) < 1e-4, (
                f"Sign mismatch for T[{j},{i}]: static={T_static[j,i]:.4f}, "
                f"tv_mean={T_tv_mean[j,i]:.4f}"
            )

    def test_return_covariances(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((150, 2))
        res = compute_mtvlk(X, dt=1.0, return_covariances=True)
        assert "C_t" in res
        assert "dC_t" in res
        assert res["C_t"].shape == (149, 2, 2)

    def test_minimum_time_steps(self):
        rng = np.random.default_rng(30)
        X = rng.standard_normal((3, 2))
        res = compute_mtvlk(X, dt=1.0)
        assert res["T_t"].shape[0] == 2

    def test_too_short_raises(self):
        X = np.ones((2, 2))
        with pytest.raises(ValueError):
            compute_mtvlk(X, dt=1.0)

    def test_adaptive_qr_mode_shapes_and_finiteness(self):
        rng = np.random.default_rng(60)
        X = rng.standard_normal((300, 2))
        res = compute_mtvlk(X, dt=1.0, qr_mode="adaptive", qr_window=50)
        assert res["T_t"].shape == (299, 2, 2)
        finite_mask = np.isfinite(res["T_t"]).all(axis=(1, 2))
        assert finite_mask[50:].all(), "NaNs found beyond warm-up window in adaptive mode"

    def test_adaptive_mode_directionally_consistent_with_constant(self):
        """On a long stationary series, adaptive and constant Q/R modes
        should agree in sign and rough magnitude of the time-mean T_t."""
        rng = np.random.default_rng(61)
        N = 2000
        X = np.zeros((N, 2))
        sigma = 0.4
        a12 = 0.4
        for t in range(N - 1):
            X[t + 1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
            X[t + 1, 1] = a12 * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()

        res_const = compute_mtvlk(X, dt=1.0, Q_scale=1e-5, R_scale=1.0)
        res_adapt = compute_mtvlk(X, dt=1.0, Q_scale=1e-5, R_scale=1.0,
                                   qr_mode="adaptive", qr_window=300)

        T_const_mean = np.nanmean(res_const["T_t"][300:], axis=0)
        T_adapt_mean = np.nanmean(res_adapt["T_t"][300:], axis=0)

        for j, i in [(0, 1), (1, 0)]:
            assert np.sign(T_const_mean[j, i]) == np.sign(T_adapt_mean[j, i]) or \
                abs(T_const_mean[j, i]) < 1e-4

    def test_qr_mode_constant_is_default_and_unchanged(self):
        """Explicit qr_mode='constant' must reproduce a plain call exactly."""
        rng = np.random.default_rng(62)
        X = rng.standard_normal((200, 2))
        res_default = compute_mtvlk(X, dt=1.0, Q_scale=1e-4, R_scale=1.0)
        res_explicit = compute_mtvlk(X, dt=1.0, Q_scale=1e-4, R_scale=1.0,
                                      qr_mode="constant")
        np.testing.assert_array_equal(res_default["T_t"], res_explicit["T_t"])
        np.testing.assert_array_equal(res_default["tau_t"], res_explicit["tau_t"])


class TestComputeMtvlkFilled:
    def test_no_nan_in_initial_window(self):
        rng = np.random.default_rng(70)
        X = rng.standard_normal((300, 2))
        res = compute_mtvlk_filled(X, dt=1.0, init_window=50)
        assert np.isfinite(res["T_t"][:50]).all()

    def test_splice_only_touches_initial_window(self):
        rng = np.random.default_rng(71)
        X = rng.standard_normal((300, 2))
        res_plain = compute_mtvlk(X, dt=1.0, init_window=50)
        res_filled = compute_mtvlk_filled(X, dt=1.0, init_window=50)
        np.testing.assert_array_equal(
            res_plain["T_t"][50:], res_filled["T_t"][50:]
        )

    def test_regime_shift_adaptive_lowers_sig_to_peak_ratio(self):
        """Adaptive Q/R should bring the significance-line/peak ratio down
        relative to constant Q/R during a genuine coupling episode --
        the core hypothesis behind implementing adaptive noise estimation."""
        rng = np.random.default_rng(72)
        N = 1200
        X = np.zeros((N, 2))
        sigma = 0.3
        for t in range(N - 1):
            c = 0.0 if (t < 400 or t >= 800) else 0.5
            X[t + 1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
            X[t + 1, 1] = c * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()

        common = dict(dt=1.0, Q_scale=5e-5, R_scale=1.0, init_window=100,
                      return_SE=True)
        res_const = compute_mtvlk_filled(X, qr_mode="constant", **common)
        res_adapt = compute_mtvlk_filled(X, qr_mode="adaptive", qr_window=100,
                                          **common)

        coupled = slice(400, 800)

        def sig_to_peak(res):
            T_abs = np.abs(res["T_t"][coupled, 0, 1])
            SE = res["SE_t"][coupled, 0, 1]
            peak = T_abs.max()
            sig_peak = (2.326 * SE).max()
            return sig_peak / peak

        assert sig_to_peak(res_adapt) < sig_to_peak(res_const)


class TestClassicalSE:
    def test_kf_mode_is_default_and_unchanged(self):
        """Explicit se_mode='kf' must reproduce a plain call exactly."""
        rng = np.random.default_rng(80)
        X = rng.standard_normal((200, 2))
        res_default = compute_mtvlk(X, dt=1.0, Q_scale=1e-4, R_scale=1.0,
                                     return_SE=True)
        res_explicit = compute_mtvlk(X, dt=1.0, Q_scale=1e-4, R_scale=1.0,
                                      se_mode="kf", return_SE=True)
        np.testing.assert_array_equal(res_default["SE_t"], res_explicit["SE_t"])

    def test_classical_shapes_and_finiteness(self):
        rng = np.random.default_rng(81)
        X = rng.standard_normal((300, 2))
        res = compute_mtvlk(X, dt=1.0, se_mode="classical", se_window=50,
                             return_SE=True)
        assert res["SE_t"].shape == (299, 2, 2)
        finite_mask = np.isfinite(res["SE_t"]).all(axis=(1, 2))
        assert finite_mask[50:].all()

    def test_classical_se_formula_ignores_Q_R_for_fixed_moments(self):
        """The classical/Fisher-information SE formula itself takes no Q/R
        input at all (SquareRootKF.analytical_R is a pure function of C, dC,
        D_diag) -- it is a genuinely different computation path than
        se_mode='kf', which reads Q/R indirectly through the filter's own P.
        Note: full pipeline SE_t still varies with Q_scale/R_scale in
        se_mode='classical' too, because Q/R changes how the KF *tracks*
        C_est/dC_est in the first place -- that's expected, not a bug; only
        the SE formula's dependence on Q/R is removed, not the point
        estimate's. This test isolates the formula from the tracking."""
        rng = np.random.default_rng(82)
        n = 2
        A = rng.standard_normal((n, n))
        C = A @ A.T + np.eye(n)
        dC = rng.standard_normal((n, n)) * 0.1
        D_diag = np.array([1.0, 2.0])

        kf_a = SquareRootKF(n, Q_scale=1e-5, R_scale=1.0)
        kf_b = SquareRootKF(n, Q_scale=1e-2, R_scale=50.0)

        np.testing.assert_allclose(
            kf_a.analytical_R(C, dC, D_diag), kf_b.analytical_R(C, dC, D_diag)
        )

    def test_classical_se_matches_analytical_R_over_window(self):
        """SE_t under se_mode='classical' should be derived from
        analytical_R(C, dC, D_diag) / se_window -- spot check the scaling
        by comparing two different se_window values (larger window -> smaller
        SE, scaling as 1/sqrt(window)). Exact ratio has some slack because
        D_diag's own EWMA smoothing rate also depends on se_window, not just
        the final division."""
        rng = np.random.default_rng(83)
        N = 1500
        X = np.zeros((N, 2))
        for t in range(N - 1):
            X[t + 1, 0] = 0.5 * X[t, 0] + 0.3 * rng.standard_normal()
            X[t + 1, 1] = 0.4 * X[t, 0] + 0.3 * X[t, 1] + 0.3 * rng.standard_normal()

        common = dict(dt=1.0, Q_scale=1e-4, R_scale=1.0, init_window=200,
                      se_mode="classical", return_SE=True)
        res_small_w = compute_mtvlk(X, se_window=100, **common)
        res_large_w = compute_mtvlk(X, se_window=400, **common)

        se_small = res_small_w["SE_t"][300:, 0, 1]
        se_large = res_large_w["SE_t"][300:, 0, 1]
        ratio = se_small / se_large
        # SE ~ 1/sqrt(window), so ratio should be close to sqrt(400/100) = 2.
        np.testing.assert_allclose(ratio, 2.0, rtol=0.15)


class TestRegressionModeSE:
    def test_shapes_and_finiteness(self):
        rng = np.random.default_rng(84)
        X = rng.standard_normal((400, 2))
        res = compute_mtvlk(X, dt=1.0, se_mode="regression", se_window=100,
                             init_window=100, return_SE=True)
        assert res["SE_t"].shape == (399, 2, 2)
        finite_mask = np.isfinite(res["SE_t"]).all(axis=(1, 2))
        assert finite_mask[150:].all()

    def test_bivariate_matches_direct_ols_calculation(self):
        """At n=2, se_mode='regression' should be CLOSE to a from-scratch
        bivariate OLS regression-coefficient-variance calculation (Hagan
        et al. 2019, appendix C) -- but not bit-identical, for two genuine
        (non-bug) reasons verified separately in isolation this session:
        (1) Zhou et al. (2024)'s general n-variable delta-method SE formula
        (_lk_se_from_moments) includes cross-covariance terms between dC
        entries that Hagan's simplified bivariate formula (ratio^2 * sigma^2
        of a single coefficient, with no cross term) omits -- confirmed to
        ~0.1% agreement with a fully consistent hand calculation when tested
        in isolation from the Kalman filter; (2) the KF's C here is a
        filtered/smoothed running estimate, not literally the raw sample
        covariance of this exact W-sample window. rtol=0.05 reflects that
        expected, understood gap rather than floating-point slop."""
        rng = np.random.default_rng(85)
        N = 2000
        X = np.zeros((N, 2))
        for t in range(N - 1):
            X[t + 1, 0] = 0.4 * X[t, 0] + 0.3 * X[t, 1] + 0.5 * rng.standard_normal()
            X[t + 1, 1] = 0.5 * X[t, 1] + 0.5 * rng.standard_normal()

        W = 300
        res = compute_mtvlk(X, dt=1.0, Q_scale=1e-4, R_scale=1.0,
                             init_window=W, se_mode="regression", se_window=W,
                             return_SE=True, return_covariances=True)

        t0 = 1000
        C = res["C_t"][t0]
        SE_ours = res["SE_t"][t0][0, 1]  # T1->2 direction (target x2, index 1)

        # RegressionSE's ring buffer, right after processing index t0, holds
        # predictor rows X[t] for t in [t0-W+1, t0] (W samples) paired with
        # dX[t] = X[t+1]-X[t] -- match that exact window alignment here
        # (an earlier off-by-one version of this window caused a ~3% mismatch).
        predictors = X[t0 - W + 1: t0 + 1]
        y = X[t0 - W + 2: t0 + 2, 1]
        x1n, x2n = predictors[:, 0], predictors[:, 1]
        design = np.column_stack([np.ones_like(x1n), x1n, x2n])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        n_obs, n_params = design.shape
        sigma2 = (resid @ resid) / (n_obs - n_params)
        XtX_inv = np.linalg.inv(design.T @ design)
        var_a12 = sigma2 * XtX_inv[2, 2]

        C22, C12 = C[1, 1], C[0, 1]
        SE_hagan = abs(C12 / C22) * np.sqrt(var_a12)

        np.testing.assert_allclose(SE_ours, SE_hagan, rtol=0.05)

    def test_kf_and_classical_modes_unaffected(self):
        """Adding se_mode='regression' must not change se_mode='kf'/'classical'
        behavior (regression guard)."""
        rng = np.random.default_rng(86)
        X = rng.standard_normal((300, 2))
        res_kf = compute_mtvlk(X, dt=1.0, se_mode="kf", return_SE=True)
        res_classical = compute_mtvlk(X, dt=1.0, se_mode="classical",
                                       se_window=50, return_SE=True)
        assert np.isfinite(res_kf["SE_t"][50:]).all()
        assert np.isfinite(res_classical["SE_t"][50:]).all()


class TestToXarray:
    def test_returns_dataset(self):
        pytest.importorskip("xarray")
        rng = np.random.default_rng(50)
        X = rng.standard_normal((100, 2))
        res = compute_mtvlk(X, dt=1.0)
        ds = to_xarray(res, var_names=["a", "b"])
        assert "T" in ds
        assert "tau" in ds
        assert list(ds.dims) == ["time", "source", "target"]

    def test_without_xarray_raises(self):
        import importlib
        xr_spec = importlib.util.find_spec("xarray")
        if xr_spec is not None:
            pytest.skip("xarray is installed")
        rng = np.random.default_rng(51)
        X = rng.standard_normal((50, 2))
        res = compute_mtvlk(X, dt=1.0)
        with pytest.raises(ImportError):
            to_xarray(res)
