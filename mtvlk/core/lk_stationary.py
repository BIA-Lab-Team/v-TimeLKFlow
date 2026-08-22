"""Static multivariate Liang-Kleeman information flow (Liang 2021).

Reference: Liang, X.S. (2021). Normalized multivariate time series causality
analysis and causal graph reconstruction. Entropy, 23(6), 679.
"""

from __future__ import annotations

import numpy as np

from mtvlk.utils.covariance import (
    finite_difference,
    sample_covariance,
    tendency_covariance,
    cofactor_matrix,
)


def _lk_from_moments(C: np.ndarray, dC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core LK formula applied to pre-computed C and dC matrices.

    Parameters
    ----------
    C  : (n, n) contemporaneous covariance, C[i,j] = cov(x_i, x_j)
    dC : (n, n) tendency cross-covariance, dC[k,i] = cov(dx_k, x_i)

    Returns
    -------
    T   : (n, n) absolute information flow, T[j,i] = T_{j->i}  (nats/time)
    tau : (n, n) relative information flow, tau[j,i] (%)
    """
    n = C.shape[0]
    det_C = np.linalg.det(C)

    if np.abs(det_C) < 1e-15:
        raise np.linalg.LinAlgError("Covariance matrix is singular; cannot compute LK flow.")

    Delta = cofactor_matrix(C)  # Delta[j,k] = det_C * inv(C)[k,j]

    # T_unnorm[j,i] = (1/det_C) * sum_k Delta[j,k] * dC[k,i]
    #               = (1/det_C) * (Delta @ dC)[j,i]
    T_unnorm = (Delta @ dC) / det_C  # [n, n]

    # T[j,i] = C[i,j]/C[i,i] * T_unnorm[j,i]
    # C.T[j,i] = C[i,j]; dividing column i by C[i,i]:
    C_ratio = C.T / np.diag(C)  # C_ratio[j,i] = C[i,j] / C[i,i]
    T = C_ratio * T_unnorm  # [n, n]

    # Noise contribution: noise_i = (dC[i,i] - T[i,i]) / C[i,i]
    # T[i,i] = T[i->i] = T_unnorm[i,i] (since C_ratio[i,i] = 1)
    diag_C = np.diag(C)
    diag_dC = np.diag(dC)
    diag_T = np.diag(T)
    noise = (diag_dC - diag_T) / diag_C  # [n]

    # Normalisation: Z_i = sum_j |T[j,i]| + |noise_i|
    Z = np.sum(np.abs(T), axis=0) + np.abs(noise)  # [n]
    Z = np.where(Z == 0, 1.0, Z)  # guard against zero
    tau = 100.0 * T / Z[np.newaxis, :]  # [n, n]

    return T, tau


def _lk_se_from_moments(
    C: np.ndarray,
    P_diag: np.ndarray,
    n_C: int,
) -> np.ndarray:
    """Analytical SE of T[j,i] from dominant dC-Jacobian terms.

    Linearisation: ∂T[j,i]/∂dC[k,i] ≈ (C[i,j]/C[i,i]) × C_inv[k,j]
    SE²[j,i] = Σ_k (∂T/∂dC[k,i])² × P_diag[n_C + k*n + i]

    Parameters
    ----------
    C      : (n, n) covariance matrix
    P_diag : (m,)  diagonal of KF state covariance P_t
    n_C    : number of C-packing elements (= n*(n+1)//2)

    Returns
    -------
    SE : (n, n) standard error of T[j,i]
    """
    n = C.shape[0]
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.zeros((n, n))
    diag_C = np.diag(C)
    P_dC = P_diag[n_C:].reshape(n, n)   # variance of dC[k,i], shape (n, n)
    SE = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if diag_C[i] < 1e-15:
                continue
            ratio = C[i, j] / diag_C[i]
            se_sq = ratio**2 * np.sum(C_inv[:, j] ** 2 * P_dC[:, i])
            SE[j, i] = np.sqrt(max(se_sq, 0.0))
    return SE


def compute_lk(
    X: np.ndarray,
    dt: float,
    n_boot: int = 200,
    block_len: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Static multivariate Liang-Kleeman information flow.

    Parameters
    ----------
    X        : (N, n) array — rows are time steps, columns are variables
    dt       : sampling interval
    n_boot   : number of block-bootstrap resamples for error estimation
    block_len: bootstrap block length; defaults to max(1, N//10)
    rng      : NumPy random Generator for reproducibility

    Returns
    -------
    dict with keys:
        T       : (n, n) absolute info flow  [T[j,i] = T_{j->i}]
        tau     : (n, n) relative info flow  [%]
        T_err   : (n, n) bootstrap std of T
        tau_err : (n, n) bootstrap std of tau
    """
    X = np.asarray(X, dtype=float)
    N, n = X.shape
    if N < 2:
        raise ValueError("Need at least 2 time steps.")

    dX = finite_difference(X, dt)  # (N-1, n)
    X_trim = X[:-1]                # align with dX

    C = sample_covariance(X_trim)
    dC = tendency_covariance(dX, X_trim)
    T, tau = _lk_from_moments(C, dC)

    if n_boot == 0:
        return {"T": T, "tau": tau, "T_err": np.zeros_like(T), "tau_err": np.zeros_like(tau)}

    rng = rng or np.random.default_rng()
    bl = block_len or max(1, (N - 1) // 10)
    T_boots = np.zeros((n_boot, n, n))
    tau_boots = np.zeros((n_boot, n, n))
    T_data = N - 1  # length of dX

    for b in range(n_boot):
        idx = _block_bootstrap_index(T_data, bl, rng)
        C_b = sample_covariance(X_trim[idx])
        dC_b = tendency_covariance(dX[idx], X_trim[idx])
        T_b, tau_b = _lk_from_moments(C_b, dC_b)
        T_boots[b] = T_b
        tau_boots[b] = tau_b

    return {
        "T": T,
        "tau": tau,
        "T_err": T_boots.std(axis=0),
        "tau_err": tau_boots.std(axis=0),
    }


def _block_bootstrap_index(T: int, bl: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a block-bootstrap index of length T with block size bl."""
    n_blocks = int(np.ceil(T / bl))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + bl, T)) for s in starts])
    return idx[:T]
