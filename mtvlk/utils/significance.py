"""Significance testing for time-varying LK information flow."""

from __future__ import annotations

import numpy as np
from scipy import stats

from mtvlk.core.mtvlk import compute_mtvlk


def pointwise_ttest(
    T_t: np.ndarray,
    X: np.ndarray,
    dt: float,
    n_perm: int = 200,
    block_len: int | None = None,
    Q_scale: float = 1e-4,
    R_scale: float = 1.0,
    init_window: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pointwise significance test for MtvLK via block-permutation null.

    At each time step, performs a two-tailed t-test comparing the observed T_t
    against a null distribution built from shuffled surrogates.

    Parameters
    ----------
    T_t      : (N-1, n, n) observed MtvLK absolute flow
    X        : (N, n) original time series
    dt       : sampling interval
    n_perm   : number of permutation surrogates
    block_len: block length for block-shuffle; defaults to max(1, N//10)
    Q_scale, R_scale, init_window : forwarded to compute_mtvlk
    rng      : NumPy random Generator

    Returns
    -------
    p_values : (n, n) array — one p-value per causal pair, testing whether
               the mean |T_{j->i}(t)| differs from the permutation null mean.
    """
    N, n = X.shape
    rng = rng or np.random.default_rng()
    bl = block_len or max(1, N // 10)

    obs_mean = np.mean(np.abs(T_t), axis=0)  # (n, n)

    null_means = np.zeros((n_perm, n, n))
    for p in range(n_perm):
        idx = _block_shuffle_index(N, bl, rng)
        X_perm = X[idx]
        res = compute_mtvlk(X_perm, dt, Q_scale=Q_scale, R_scale=R_scale,
                            init_window=init_window)
        null_means[p] = np.mean(np.abs(res["T_t"]), axis=0)

    # Two-tailed t-test: obs vs null distribution
    null_mean_m = null_means.mean(axis=0)
    null_std_m = null_means.std(axis=0, ddof=1)
    t_stat = (obs_mean - null_mean_m) / (null_std_m / np.sqrt(n_perm) + 1e-30)
    p_values = 2 * stats.t.sf(np.abs(t_stat), df=n_perm - 1)
    return p_values


def _block_shuffle_index(N: int, bl: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole blocks within the time axis, preserving short-range autocorr."""
    n_blocks = int(np.ceil(N / bl))
    starts = np.arange(0, N, bl)
    rng.shuffle(starts)
    idx = np.concatenate([np.arange(s, min(s + bl, N)) for s in starts])
    return idx[:N]
