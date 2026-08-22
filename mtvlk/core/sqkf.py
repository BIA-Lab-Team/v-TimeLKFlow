"""Square-Root Kalman Filter for tracking time-varying second-order moments.

State vector θ_t packs:
  - n(n+1)/2 contemporaneous covariance elements C[k,i] = cov(x_k, x_i), k <= i
  - n^2 tendency cross-covariance elements dC[k,i] = cov(x_k, ẋ_i), all k,i

Observation model: y_t = θ_t + ε_t  (H = I, linear)
State transition:  θ_t = θ_{t-1} + η_t  (random walk, Φ = I)

Covariance factors are propagated via Cholesky (S.T @ S = P) for numerical
stability with near-singular matrices.
"""

from __future__ import annotations

import numpy as np


class SquareRootKF:
    """Kalman Filter (Cholesky square-root form) for time-varying LK moments.

    Parameters
    ----------
    n       : number of variables
    Q_scale : process noise variance scale (scalar × I)
    R_scale : observation noise variance scale (scalar × I)
    """

    def __init__(self, n: int, Q_scale: float = 1e-4, R_scale: float = 1.0) -> None:
        self.n = n
        self.n_C = n * (n + 1) // 2
        self.n_dC = n * n
        self.m = self.n_C + self.n_dC

        self._triu_idx = np.triu_indices(n)
        self._Q = Q_scale * np.eye(self.m)
        self._R = R_scale * np.eye(self.m)

        self.theta: np.ndarray = np.zeros(self.m)
        # S is upper-triangular; P = S.T @ S
        self.S: np.ndarray = np.eye(self.m)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        X_init: np.ndarray,
        dX_init: np.ndarray,
        P0_scale: float = 1.0,
    ) -> None:
        """Seed the filter state from a warm-up window.

        Parameters
        ----------
        X_init  : (T0, n) warm-up observations
        dX_init : (T0, n) corresponding forward-difference tendencies
        P0_scale: scale of the initial state uncertainty covariance
        """
        T0 = X_init.shape[0]
        n = self.n

        if T0 < 2:
            # Fall back to identity-scaled guess; filter will correct quickly
            C_init = np.eye(n)
            dC_init = np.zeros((n, n))
        else:
            X_c = X_init - X_init.mean(axis=0)
            dX_c = dX_init - dX_init.mean(axis=0)
            C_init = (X_c.T @ X_c) / (T0 - 1)
            dC_init = (X_c.T @ dX_c) / (T0 - 1)

        self.theta = self._pack(C_init, dC_init)
        P0 = P0_scale * np.eye(self.m)
        self.S = np.linalg.cholesky(P0).T         # upper triangular

    def predict(self, Q: np.ndarray | None = None) -> None:
        """Random-walk predict: P_{t|t-1} = P_{t-1} + Q.

        Parameters
        ----------
        Q : optional per-step process noise, overriding self._Q for this
            call only. Shape (m,) (diagonal vector) or (m, m) (full matrix).
        """
        Q_mat = self._as_matrix(Q) if Q is not None else self._Q
        P = self.S.T @ self.S
        P_pred = P + Q_mat
        self.S = np.linalg.cholesky(P_pred + 1e-14 * np.eye(self.m)).T

    def update(self, y: np.ndarray, R: np.ndarray | None = None) -> None:
        """Update state given observation vector y of length m.

        Parameters
        ----------
        R : optional per-step measurement noise, overriding self._R for this
            call only. Shape (m,) (diagonal vector) or (m, m) (full matrix).
        """
        R_mat = self._as_matrix(R) if R is not None else self._R
        P = self.S.T @ self.S
        nu = y - self.theta                        # innovation (H = I)
        P_e = P + R_mat                            # innovation covariance (H = I)
        # Kalman gain: K = P H^T P_e^{-1} = P P_e^{-1} (H = I)
        K = np.linalg.solve(P_e.T, P.T).T
        self.theta = self.theta + K @ nu
        # Joseph form for P_new (numerically stable):
        IKH = np.eye(self.m) - K
        P_new = IKH @ P @ IKH.T + K @ R_mat @ K.T
        self.S = np.linalg.cholesky(P_new + 1e-14 * np.eye(self.m)).T

    def extract_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Unpack current state into C[n,n] and dC[n,n].

        Returns
        -------
        C  : (n, n) symmetric covariance estimate  C[k,i] = cov(x_k, x_i)
        dC : (n, n) tendency cross-cov estimate    dC[k,i] = cov(x_k, ẋ_i)
        """
        n = self.n
        C = np.zeros((n, n))
        C[self._triu_idx] = self.theta[: self.n_C]
        C = C + C.T - np.diag(np.diag(C))          # symmetrise
        dC = self.theta[self.n_C :].reshape(n, n)
        return C, dC

    def build_observation(
        self,
        x: np.ndarray,
        dx: np.ndarray,
        mu_x: np.ndarray,
        mu_dx: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector y_t from a single time step.

        Packs mean-centred products matching the state vector layout:
        - C part:  (x_k - mu_k) * (x_i - mu_i) for k <= i
        - dC part: (x_k - mu_k) * (dx_i - mu_dxi) for all k, i
        """
        xc = x - mu_x
        dxc = dx - mu_dx

        C_obs = xc[self._triu_idx[0]] * xc[self._triu_idx[1]]        # (n_C,)
        # dC[k,i] = xc_k * dxc_i  → outer product, row-major
        dC_obs = (xc[:, np.newaxis] * dxc[np.newaxis, :]).ravel()    # (n^2,)
        return np.concatenate([C_obs, dC_obs])

    def analytical_R(
        self,
        C: np.ndarray,
        dC: np.ndarray,
        D_diag: np.ndarray,
        R_floor: float = 1e-10,
    ) -> np.ndarray:
        """Closed-form sampling variance of a single raw observation y_t
        around its true value, for jointly Gaussian data.

        For zero-mean jointly Gaussian (a, b), Var(a*b) = Cov(a,a)*Cov(b,b)
        + Cov(a,b)**2 (Isserlis'/Wick's theorem). Applied to each packed
        observation component built by `build_observation`:

          C part:  y[k,i] = xc_k * xc_i
                   Var = C[k,k]*C[i,i] + C[k,i]**2         (k != i)
                   Var = 2*C[k,k]**2                       (k == i)
          dC part: y[k,i] = xc_k * dxc_i
                   Var = C[k,k]*D_diag[i] + dC[k,i]**2

        Parameters
        ----------
        C  : (n, n) current contemporaneous covariance estimate.
        dC : (n, n) current tendency cross-covariance estimate.
        D_diag : (n,) current estimate of Var(dx_i) for each variable
            (not tracked by the filter's own state; supplied by the caller,
            e.g. a simple running variance of the tendency series).
        R_floor : minimum variance floor.

        Returns
        -------
        R : (m,) diagonal observation noise vector, packed in the same
            order as `build_observation`'s output.
        """
        k_idx, i_idx = self._triu_idx
        C_diag = np.diag(C)
        C_kk = C_diag[k_idx]
        C_ii = C_diag[i_idx]
        C_ki = C[k_idx, i_idx]
        off_diag = C_kk * C_ii + C_ki ** 2
        on_diag = 2 * C_kk ** 2
        R_C = np.where(k_idx == i_idx, on_diag, off_diag)

        # dC part: row-major (k, i) matches build_observation's outer-product ravel.
        R_dC = (C_diag[:, np.newaxis] * D_diag[np.newaxis, :] + dC ** 2).ravel()

        R = np.concatenate([R_C, R_dC])
        return np.maximum(R, R_floor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_matrix(v: np.ndarray) -> np.ndarray:
        """Convert a diagonal vector (m,) to a full (m, m) matrix; pass
        through unchanged if already 2-D."""
        return np.diag(v) if v.ndim == 1 else v

    def _pack(self, C: np.ndarray, dC: np.ndarray) -> np.ndarray:
        """Pack C (upper-tri) and dC (all elements, row-major) into θ."""
        return np.concatenate([C[self._triu_idx], dC.ravel()])
