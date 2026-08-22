"""Low-level covariance and cofactor helpers."""

import numpy as np


def finite_difference(X: np.ndarray, dt: float) -> np.ndarray:
    """Forward finite-difference tendency: dX[t] = (X[t+1] - X[t]) / dt.

    Parameters
    ----------
    X : array of shape (N, n)
    dt : time step

    Returns
    -------
    dX : array of shape (N-1, n)
    """
    return (X[1:] - X[:-1]) / dt


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance matrix C[i,j] = cov(x_i, x_j).

    Parameters
    ----------
    X : array of shape (T, n)  — time × variables

    Returns
    -------
    C : array of shape (n, n)
    """
    return np.cov(X.T, ddof=1)


def tendency_covariance(dX: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cross-covariance dC[k,i] = cov(X_k, dX_i).

    Matches the Liang (2021) convention used in T_{j→i}: the k-th row of dC
    corresponds to x_k (contemporaneous), and the i-th column corresponds to
    ẋ_i (tendency of variable i).

    Both dX and X must have the same length T (use X[:-1] if X has one extra row).

    Parameters
    ----------
    dX : array of shape (T, n)  — forward-difference tendencies
    X  : array of shape (T, n)  — contemporaneous values

    Returns
    -------
    dC : array of shape (n, n)  — dC[k, i] = cov(X[:,k], dX[:,i])
    """
    T = dX.shape[0]
    dX_c = dX - dX.mean(axis=0)
    X_c = X - X.mean(axis=0)
    return (X_c.T @ dX_c) / (T - 1)  # [n, n]


def cofactor_matrix(C: np.ndarray) -> np.ndarray:
    """Cofactor matrix Delta where Delta[j,k] = det(C) * inv(C)[k,j].

    This equals adj(C).T, which is what the Liang (2021) formula uses.

    Parameters
    ----------
    C : symmetric positive-definite matrix of shape (n, n)

    Returns
    -------
    Delta : array of shape (n, n)
    """
    det_C = np.linalg.det(C)
    C_inv = np.linalg.inv(C)
    return det_C * C_inv.T  # Delta[j,k] = det_C * C_inv[k,j]


def pack_upper(M: np.ndarray) -> np.ndarray:
    """Pack the upper-triangular part of M (including diagonal) into a 1-D vector.

    Index order follows numpy triu_indices row-major.
    """
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx]


def unpack_upper(v: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a symmetric matrix from its packed upper-triangular vector."""
    M = np.zeros((n, n))
    idx = np.triu_indices(n)
    M[idx] = v
    M = M + M.T - np.diag(np.diag(M))
    return M
