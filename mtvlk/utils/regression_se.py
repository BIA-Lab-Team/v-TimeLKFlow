"""Windowed VAR(1) regression-coefficient variance for the significance test.

Hagan et al. (2019), appendix C, gives the paper's actual significance-test
formula: for a bivariate AR(1) fit X1(t+1) = f1 + a11*X1(t) + a12*X2(t) +
eps(t), the information flow T2->1 = (C12/C11)*a12 is asymptotically normal
with variance (C12/C11)^2 * sigma^2_a12, where sigma^2_a12 is the classical
OLS/MLE variance of the fitted regression coefficient a12 -- derived from the
Fisher information matrix of the model likelihood, i.e. exactly the standard
OLS coefficient-covariance formula sigma^2 * (X^T X)^-1.

This module generalizes that to n variables: for each target variable i, fit
a windowed VAR(1)-style regression dX_i(t) = f_i + sum_k a_ik * X_k(t) +
eps_i(t) (regressing the tendency on contemporaneous X, matching this
package's dC = cov(X, dX) convention throughout -- equivalent up to a
deterministic reparameterization to Hagan's literal X(t+1) regression, since
dX_i(t) = X_i(t+1) - X_i(t), which does not change the OLS coefficient
covariance matrix). The delta method then maps the regression-coefficient
covariance to Var(dC[j,i]), since dC[:,i] = C @ a_i in a linear system, for
use by `mtvlk.core.lk_stationary._lk_se_from_moments`.

This is a more literal implementation of Hagan/Zhou's appendix C than
`mtvlk.core.sqkf.SquareRootKF.analytical_R` (which approximates the same
underlying idea via the closed-form Gaussian product-moment variance of a
windowed *sample covariance*, rather than a fitted *regression coefficient*
-- the two are close in practice but not algebraically identical).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RegressionSE:
    """Online, causal estimator of Var(dC[j,i]) via a windowed VAR(1)
    regression fit, following Hagan et al. (2019) appendix C.

    Parameters
    ----------
    n : number of variables.
    window : number of most-recent samples used for the regression fit.
    """

    def __init__(self, n: int, window: int) -> None:
        self.n = n
        self.window = window
        p = n + 1  # intercept + n predictors

        self._buf: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=window)
        self._Sxx = np.zeros((p, p))
        self._Sxy = np.zeros((p, n))
        self._Syy_diag = np.zeros(n)

    def warm_start(self, X_batch: np.ndarray, dX_batch: np.ndarray) -> None:
        """Preload the running sums from a batch of past observations (e.g.
        the same warm-up window used to seed the Kalman filter)."""
        for x, dx in zip(X_batch, dX_batch):
            self.update(x, dx)

    def update(self, x: np.ndarray, dx: np.ndarray) -> None:
        """Push this step's (x(t), dx(t)) sample into the running sums,
        evicting the oldest sample if already at capacity."""
        design = np.concatenate([[1.0], x])  # (n+1,)

        if len(self._buf) == self.window:
            old_design, old_dx = self._buf[0]
            self._Sxx -= np.outer(old_design, old_design)
            self._Sxy -= np.outer(old_design, old_dx)
            self._Syy_diag -= old_dx ** 2

        self._buf.append((design, dx))
        self._Sxx += np.outer(design, design)
        self._Sxy += np.outer(design, dx)
        self._Syy_diag += dx ** 2

    def variance(self, C: np.ndarray) -> np.ndarray | None:
        """Return the (n, n) Var(dC[j, i]) matrix given the current
        contemporaneous covariance estimate `C` (used as the delta-method
        Jacobian: dC[:, i] = C @ a_i for the i-th equation's coefficients).

        Returns None if there are not yet enough samples to fit (count must
        exceed n+1 degrees of freedom).
        """
        n = self.n
        p = n + 1
        count = len(self._buf)
        dof = count - p
        if dof <= 0:
            return None

        try:
            XtX_inv = np.linalg.inv(self._Sxx)
        except np.linalg.LinAlgError:
            return None

        beta = XtX_inv @ self._Sxy               # (p, n): beta[:, i] for target i
        resid_ss = self._Syy_diag - np.einsum("pi,pi->i", beta, self._Sxy)
        sigma2 = np.maximum(resid_ss, 0.0) / dof  # (n,)

        A_inv = XtX_inv[1:, 1:]                   # (n, n): covariance shape for the slopes
        Var_dC = np.zeros((n, n))
        for i in range(n):
            Cov_a_i = sigma2[i] * A_inv
            Var_dC[:, i] = np.einsum("jk,kl,jl->j", C, Cov_a_i, C)
        return Var_dC
