"""Online EWMA estimation of time-varying Kalman filter process noise Q_t.

Zhou et al. (2024), Sec. 2b, states that Q (process noise) and R (measurement
noise) are not assumed constant, and are instead estimated "offline" via an
exponentially weighted moving average (EWMA) and an unweighted moving average
(UWMA) over a lookback window, whose results are then fed into the Kalman
filter recursion (their Eqs. 12-13). The paper does not give an explicit
closed-form formula for how EWMA/UWMA are applied to obtain Q and R.

This module handles Q_t only. R_t is instead derived analytically from the
filter's own current covariance/tendency-covariance estimate -- see
`SquareRootKF.analytical_R` -- rather than estimated empirically from the raw
observation series. Two empirical R_t reconstructions were tried here first
(variance of y_t around its own rolling mean, then a successive-difference
noise floor) and both left the significance threshold poorly scaled relative
to the paper's actual Figure 2/3 (confirmed by comparing against the
rendered paper figure: the paper's significance line sits at roughly 1/6-1/3
of the peak |T|, both empirical R_t reconstructions gave a much larger
ratio). A single raw observation y_t = (x_k-mu_k)(x_i-mu_i) (or the
tendency-cross equivalent) is a one-sample estimator of a covariance entry;
for jointly Gaussian data its sampling variance around the *true* covariance
has a known closed form, which is what `analytical_R` uses instead of an
empirical reconstruction.

Q_t (process noise, i.e. how fast the true covariance structure drifts) has
no equivalent closed form -- it is a genuine modeling choice about the
state's random-walk dynamics -- so it is still estimated empirically here:
an EWMA of the squared step-to-step change in a rolling mean of the raw
observation series y_t.

This is a defensible reconstruction grounded in standard windowed noise
estimation and the closed-form Gaussian product-moment variance, not a
verified match to the paper authors' exact method.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def ewma_update(prev: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average update: alpha*x + (1-alpha)*prev."""
    return alpha * x + (1.0 - alpha) * prev


def window_to_alpha(window: int) -> float:
    """Standard EWMA<->window-length equivalence: alpha = 2/(window+1)."""
    return 2.0 / (window + 1)


class AdaptiveQR:
    """Online, causal estimator of diagonal Q_t (process noise), driven
    purely by the raw observation series y_t.

    Parameters
    ----------
    m : observation/state dimension
    window : lookback window length W for the UWMA rolling mean of y_t, and,
        unless `ewma_alpha` is given, for the EWMA smoothing factor applied
        to the rolling mean's increments, via `window_to_alpha(window)`.
    ewma_alpha : optional override for Q_t's EWMA smoothing factor.
    Q_floor : minimum variance floor, to keep the filter's covariance matrix
        positive definite even if an estimated variance collapses toward
        zero.
    Q_scale : initial Q_t value used before any observations have been
        pushed through `update`.
    """

    def __init__(
        self,
        m: int,
        window: int,
        ewma_alpha: float | None = None,
        Q_floor: float = 1e-10,
        Q_scale: float = 1e-4,
    ) -> None:
        self.m = m
        self.window = window
        self.alpha = ewma_alpha if ewma_alpha is not None else window_to_alpha(window)
        self.Q_floor = Q_floor

        self.Q_t = np.full(m, Q_scale)

        # Ring buffer of y, plus running sum, for O(1) rolling mean.
        self._y_buf: deque[np.ndarray] = deque(maxlen=window)
        self._y_sum = np.zeros(m)
        self._prev_roll_mean: np.ndarray | None = None

    def warm_start(self, y_batch: np.ndarray) -> None:
        """Preload the rolling buffer from a batch of past observations
        (e.g. the same warm-up window used to seed the Kalman filter),
        so the estimator starts with a full window instead of a single
        noisy sample dominating Q_t at the first live step.
        """
        for y in y_batch:
            self.update(y)

    def update(self, y: np.ndarray) -> np.ndarray:
        """Push this step's raw observation `y` (shape (m,)) into the
        running estimator, and return the Q_t vector to use for the *next*
        filter step.
        """
        if len(self._y_buf) == self.window:
            self._y_sum -= self._y_buf[0]
        self._y_buf.append(y)
        self._y_sum += y

        roll_mean = self._y_sum / len(self._y_buf)
        if self._prev_roll_mean is not None:
            drift_sq = (roll_mean - self._prev_roll_mean) ** 2
            self.Q_t = ewma_update(self.Q_t, drift_sq, self.alpha)
        self._prev_roll_mean = roll_mean

        self.Q_t = np.maximum(self.Q_t, self.Q_floor)
        return self.Q_t
