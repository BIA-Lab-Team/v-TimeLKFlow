"""Tests for the EWMA process-noise (Q_t) estimator."""

import numpy as np

from mtvlk.utils.adaptive_noise import AdaptiveQR, ewma_update, window_to_alpha


class TestEwmaUpdate:
    def test_basic_step(self):
        prev = np.array([1.0, 2.0])
        x = np.array([3.0, 4.0])
        out = ewma_update(prev, x, alpha=0.5)
        np.testing.assert_allclose(out, [2.0, 3.0])

    def test_alpha_one_returns_x(self):
        prev = np.array([1.0])
        x = np.array([9.0])
        out = ewma_update(prev, x, alpha=1.0)
        np.testing.assert_allclose(out, [9.0])

    def test_alpha_zero_returns_prev(self):
        prev = np.array([1.0])
        x = np.array([9.0])
        out = ewma_update(prev, x, alpha=0.0)
        np.testing.assert_allclose(out, [1.0])


class TestWindowToAlpha:
    def test_known_value(self):
        assert window_to_alpha(299) == 2.0 / 300

    def test_monotonic_decreasing_in_window(self):
        alphas = [window_to_alpha(w) for w in (10, 100, 1000)]
        assert alphas[0] > alphas[1] > alphas[2]


class TestAdaptiveQR:
    def test_positive_and_finite(self):
        rng = np.random.default_rng(0)
        m = 5
        est = AdaptiveQR(m, window=20)
        for _ in range(200):
            y = rng.standard_normal(m)
            Q_t = est.update(y)
            assert np.all(np.isfinite(Q_t))
            assert np.all(Q_t > 0)

    def test_q_reacts_to_level_shift(self):
        """A slow drift in the mean of y should raise Q_t (the estimated
        process/drift noise)."""
        rng = np.random.default_rng(3)
        m = 1
        window = 50
        est = AdaptiveQR(m, window=window)

        for _ in range(200):
            y = rng.standard_normal(m) * 0.1
            Q_t = est.update(y)
        Q_before = Q_t.copy()

        # A steady upward drift in the mean level (not just noise).
        level = 0.0
        for _ in range(200):
            level += 0.05
            y = np.array([level]) + rng.standard_normal(m) * 0.1
            Q_t = est.update(y)

        assert Q_t[0] > Q_before[0]
        assert np.all(np.isfinite(Q_t))
        assert Q_t[0] < 1e6  # no blow-up

    def test_warm_start_avoids_cold_start_dominance(self):
        """After warm_start on a steady batch, the first live sample should
        not fully overwrite Q_t (buffer should already be at window size)."""
        rng = np.random.default_rng(5)
        m = 1
        window = 50
        est = AdaptiveQR(m, window=window)
        steady_batch = rng.standard_normal((window, m)) * 0.1
        est.warm_start(steady_batch)

        Q_before = est.Q_t.copy()
        shock = np.array([100.0])
        Q_after = est.update(shock)

        # A single extreme sample shouldn't dominate Q_t after a full
        # warm-started window (EWMA weight ~= 2/(window+1), much less than 1).
        assert Q_after[0] < 0.5 * shock[0] ** 2
        assert Q_after[0] != Q_before[0]
