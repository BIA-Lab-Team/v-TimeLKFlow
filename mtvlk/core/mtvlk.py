"""Time-varying multivariate Liang-Kleeman information flow (Zhou et al. 2024).

Extends the static multivariate LK (Liang 2021) to non-stationary systems by
replacing the fixed sample covariances with time-varying estimates from a
Square-Root Kalman Filter.

Reference: Zhou et al. (2024). Estimating time-dependent structures in a
multivariate causality for land-atmosphere interactions. J. Climate, 37(6).
DOI: 10.1175/JCLI-D-23-0207.1
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from mtvlk.core.lk_stationary import _lk_from_moments, _lk_se_from_moments
from mtvlk.core.sqkf import SquareRootKF
from mtvlk.utils.adaptive_noise import AdaptiveQR, ewma_update, window_to_alpha
from mtvlk.utils.covariance import finite_difference
from mtvlk.utils.regression_se import RegressionSE


def _run_forward_pass(
    X: np.ndarray,
    dt: float,
    Q_scale: float,
    R_scale: float,
    init_window: int | None,
    qr_mode: Literal["constant", "adaptive"],
    qr_window: int | None,
    ewma_alpha: float | None,
    return_covariances: bool,
    return_SE: bool,
    kf_state_init: tuple[np.ndarray, np.ndarray] | None = None,
    mu_state_init: tuple[np.ndarray, np.ndarray, int] | None = None,
    se_mode: Literal["kf", "classical", "regression"] = "kf",
    se_window: int | None = None,
) -> dict:
    """Run one forward-in-time MtvLK pass over `X`.

    If `kf_state_init` is given, the filter is warm-started directly from
    that (theta, S) state instead of calling `kf.initialize` on a warm-up
    window of `X` -- used by `compute_mtvlk_filled`'s backward pass, which
    is seeded from the forward pass's final state (Zhou et al. 2024,
    appendix A).

    Returns a dict with the same array keys as `compute_mtvlk`'s result,
    plus "final_theta", "final_S", "final_mu_x", "final_mu_dx",
    "final_count", and "w" (the warm-up length consumed by this pass, 0 if
    `kf_state_init` was supplied).
    """
    X = np.asarray(X, dtype=float)
    N, n = X.shape
    if N < 3:
        raise ValueError("Need at least 3 time steps.")

    dX = finite_difference(X, dt)  # (N-1, n)
    T_len = N - 1                  # number of usable time steps

    kf = SquareRootKF(n, Q_scale=Q_scale, R_scale=R_scale)

    if kf_state_init is None:
        w = init_window if init_window is not None else max(2 * n * n, 30)
        w = min(w, T_len - 1)     # must leave at least one step for the filter
        # Start P near its steady-state value to avoid a large initial SE transient.
        P0_scale = float(np.sqrt(Q_scale * R_scale))
        kf.initialize(X[:w], dX[:w], P0_scale=P0_scale)
        mu_x = X[:w].mean(axis=0)
        mu_dx = dX[:w].mean(axis=0)
        count = w
    else:
        w = 0
        kf.theta, kf.S = kf_state_init
        mu_x, mu_dx, count = mu_state_init

    default_window = init_window or max(2 * n * n, 30)
    need_D_diag = qr_mode == "adaptive" or se_mode == "classical"

    qr_est = None
    Q_t = Q_scale * np.ones(kf.m)
    D_diag = np.ones(n)  # running Var(dx_i), used by the analytical R_t formula
    if need_D_diag and w > 0:
        D_diag = dX[:w].var(axis=0)

    if qr_mode == "adaptive":
        window = qr_window if qr_window is not None else default_window
        qr_est = AdaptiveQR(kf.m, window, ewma_alpha=ewma_alpha, Q_scale=Q_scale)
        if w > 0:
            # Warm-start the Q_t estimator's rolling buffer from the same
            # warm-up window used to seed the filter (using the batch mean,
            # not the noisy KF-replay path), so it doesn't start cold with a
            # single sample dominating Q_t at the first live step.
            y_warm = np.stack([
                kf.build_observation(X[i], dX[i], mu_x, mu_dx) for i in range(w)
            ])
            qr_est.warm_start(y_warm)
        Q_t = qr_est.Q_t

    se_window_val = se_window if se_window is not None else (qr_window if qr_window is not None else default_window)
    alpha_D = qr_est.alpha if qr_est is not None else window_to_alpha(se_window_val)

    reg_se = None
    if se_mode == "regression":
        reg_se = RegressionSE(n, se_window_val)
        if w > 0:
            reg_se.warm_start(X[:w], dX[:w])

    # Output arrays
    T_t = np.full((T_len, n, n), np.nan)
    tau_t = np.full((T_len, n, n), np.nan)
    C_t_arr = np.full((T_len, n, n), np.nan) if return_covariances else None
    dC_t_arr = np.full((T_len, n, n), np.nan) if return_covariances else None
    SE_t_arr = np.full((T_len, n, n), np.nan) if return_SE else None

    # --- forward pass ---
    for t in range(T_len):
        if qr_mode == "adaptive":
            kf.predict(Q_t)
            # Analytical R_t: closed-form sampling variance of a single raw
            # observation around the filter's current (predicted) moment
            # estimate, rather than an empirical reconstruction (see
            # mtvlk.utils.adaptive_noise module docstring for why).
            C_prior, dC_prior = kf.extract_moments()
            R_t = kf.analytical_R(C_prior, dC_prior, D_diag)
        else:
            kf.predict()

        # Build observation from current x(t) and dx(t)
        y_t = kf.build_observation(X[t], dX[t], mu_x, mu_dx)

        if qr_mode == "adaptive":
            kf.update(y_t, R_t)
        else:
            kf.update(y_t)

        # Skip feeding the reprocessed warm-up window (t < w) into the
        # adaptive/D_diag estimators: those samples were already
        # batch-averaged into the filter's initial seed and are noisier on
        # replay, so letting them sit in the rolling buffer causes a
        # "hangover" artifact for roughly one more window length after t = w.
        if t >= w:
            if qr_mode == "adaptive":
                Q_t = qr_est.update(y_t)
            if need_D_diag:
                dx_c = dX[t] - mu_dx
                D_diag = ewma_update(D_diag, dx_c ** 2, alpha_D)
            if reg_se is not None:
                reg_se.update(X[t], dX[t])

        # Update running mean (expanding window, causal)
        count += 1
        mu_x = mu_x + (X[t] - mu_x) / count
        mu_dx = mu_dx + (dX[t] - mu_dx) / count

        C_est, dC_est = kf.extract_moments()

        if se_mode == "classical":
            # Fisher-information/classical variance of a windowed sample
            # covariance estimate (Hagan et al. 2019, appendix C; Zhou et al.
            # 2024, appendix B), decoupled from the KF's own Q/R -- see
            # SquareRootKF.analytical_R's docstring for the underlying
            # closed-form Gaussian product-moment variance this reuses.
            P_diag = kf.analytical_R(C_est, dC_est, D_diag) / se_window_val
        elif se_mode == "regression":
            # Literal Hagan et al. (2019) appendix C formula: variance of a
            # windowed VAR(1) regression coefficient, mapped to Var(dC) via
            # the delta method -- see RegressionSE's docstring.
            Var_dC = reg_se.variance(C_est)
            P_diag = (np.concatenate([np.zeros(kf.n_C), Var_dC.ravel()])
                      if Var_dC is not None else np.zeros(kf.m))
        else:
            # diagonal of state covariance P_t = S.T @ S
            P_diag = np.sum(kf.S ** 2, axis=0)

        # Guard against ill-conditioned covariance
        try:
            T_step, tau_step = _lk_from_moments(C_est, dC_est)
            SE_step = _lk_se_from_moments(C_est, P_diag, kf.n_C) if return_SE else None
        except np.linalg.LinAlgError:
            T_step = np.zeros((n, n))
            tau_step = np.zeros((n, n))
            SE_step = np.zeros((n, n)) if return_SE else None

        T_t[t] = T_step
        tau_t[t] = tau_step

        if return_covariances:
            C_t_arr[t] = C_est
            dC_t_arr[t] = dC_est
        if return_SE and SE_step is not None:
            SE_t_arr[t] = SE_step

    result = {
        "T_t": T_t,
        "tau_t": tau_t,
        "time_idx": np.arange(T_len),
    }
    if return_covariances:
        result["C_t"] = C_t_arr
        result["dC_t"] = dC_t_arr
    if return_SE:
        result["SE_t"] = SE_t_arr

    return {
        "result": result,
        "final_theta": kf.theta.copy(),
        "final_S": kf.S.copy(),
        "final_mu_x": mu_x.copy(),
        "final_mu_dx": mu_dx.copy(),
        "final_count": count,
        "w": w,
    }


def compute_mtvlk(
    X: np.ndarray,
    dt: float,
    Q_scale: float = 1e-4,
    R_scale: float = 1.0,
    init_window: int | None = None,
    qr_mode: Literal["constant", "adaptive"] = "constant",
    qr_window: int | None = None,
    ewma_alpha: float | None = None,
    se_mode: Literal["kf", "classical", "regression"] = "kf",
    se_window: int | None = None,
    return_covariances: bool = False,
    return_SE: bool = False,
) -> dict:
    """Time-varying multivariate Liang-Kleeman information flow (MtvLK).

    Parameters
    ----------
    X       : (N, n) array — rows are time steps, columns are variables
    dt      : sampling interval
    Q_scale : process noise scale for the Square-Root Kalman Filter.
              Controls how fast covariances can evolve. Larger = faster tracking.
              In `qr_mode="adaptive"`, used only as the initial Q_t value.
    R_scale : observation noise scale. Larger = smoother (less reactive) estimates.
              In `qr_mode="adaptive"`, used only as the initial R_t value.
    init_window : number of initial time steps used to seed the KF state.
              Defaults to max(2*n^2, 30).
    qr_mode : "constant" (default, original behavior) keeps Q/R fixed at
              Q_scale/R_scale for the whole run. "adaptive" estimates
              time-varying Q_t/R_t online via `mtvlk.utils.adaptive_noise.
              AdaptiveQR` (Zhou et al. 2024, Sec. 2b describes Q/R as
              estimated via EWMA/UWMA rather than held constant; see that
              module's docstring for the reconstruction used here).
    qr_window : lookback window for the adaptive Q/R estimator. Defaults to
              `init_window` (or its own default) if not given.
    ewma_alpha : optional override for the adaptive estimator's Q_t EWMA
              smoothing factor. Ignored when qr_mode="constant".
    se_mode : "kf" (default, original behavior) derives SE_t from the Kalman
              filter's own state covariance P, which depends on whatever
              Q_scale/R_scale (or adaptive Q_t/R_t) was chosen. "classical"
              instead computes SE_t from the closed-form Gaussian
              product-moment sampling variance of a `se_window`-sample
              covariance estimate (Hagan et al. 2019, appendix C; Zhou et al.
              2024, appendix B use a Fisher-information/MLE variance of a
              local AR-model fit, which for Gaussian data is the same
              quantity) -- entirely independent of Q/R. See
              `SquareRootKF.analytical_R`'s docstring for the formula reused.
    se_window : effective sample size for `se_mode="classical"`. Defaults to
              `qr_window`, then `init_window` (or its own default).
    return_covariances : if True, also return the time-varying C_t and dC_t arrays.

    Returns
    -------
    dict with keys:
        T_t   : (N-1, n, n) absolute info flow at each time step [T_t[t,j,i] = T_{j->i}(t)]
        tau_t : (N-1, n, n) relative info flow [%]
        time_idx : (N-1,) integer indices into the original N-length time axis
        C_t   : (N-1, n, n) estimated covariance at each step  [if return_covariances]
        dC_t  : (N-1, n, n) estimated tendency covariance       [if return_covariances]
        SE_t  : (N-1, n, n) analytical SE of T_t               [if return_SE]
    """
    pass_out = _run_forward_pass(
        X, dt, Q_scale, R_scale, init_window,
        qr_mode, qr_window, ewma_alpha,
        return_covariances, return_SE,
        se_mode=se_mode, se_window=se_window,
    )
    return pass_out["result"]


def compute_mtvlk_filled(
    X: np.ndarray,
    dt: float,
    Q_scale: float = 1e-4,
    R_scale: float = 1.0,
    init_window: int | None = None,
    qr_mode: Literal["constant", "adaptive"] = "constant",
    qr_window: int | None = None,
    ewma_alpha: float | None = None,
    se_mode: Literal["kf", "classical", "regression"] = "kf",
    se_window: int | None = None,
    fill_window: int | None = None,
    return_covariances: bool = False,
    return_SE: bool = False,
) -> dict:
    """MtvLK with the initial lookback-window gap filled via a backward pass
    (Zhou et al. 2024, appendix A).

    A plain forward `compute_mtvlk` pass needs `init_window` time steps to
    seed the Kalman filter, so its estimate is unreliable there. Appendix A
    of the paper addresses this by: reversing the input series in time,
    running the same MtvLK pass on the reversed series (warm-started from
    the *forward* pass's final filter state, per the paper's description),
    reversing that result back into forward-time order, and splicing it into
    the forward pass's initial window.

    Note (carried over from the paper's own appendix A): this splice is a
    numerical patch for the missing initial window, not a causally-justified
    estimate of what happens there -- it is computed from data run in
    reverse.

    Parameters are identical to `compute_mtvlk` (including `se_mode`/
    `se_window`), plus:

    fill_window : number of initial time steps to splice in from the
              backward pass. Defaults to the forward pass's warm-up length
              (`init_window`, or its own default).

    Returns
    -------
    Same dict shape as `compute_mtvlk`.
    """
    fwd = _run_forward_pass(
        X, dt, Q_scale, R_scale, init_window,
        qr_mode, qr_window, ewma_alpha,
        return_covariances, return_SE,
        se_mode=se_mode, se_window=se_window,
    )

    X_rev = np.asarray(X, dtype=float)[::-1].copy()
    bwd = _run_forward_pass(
        X_rev, dt, Q_scale, R_scale, init_window,
        qr_mode, qr_window, ewma_alpha,
        return_covariances, return_SE,
        kf_state_init=(fwd["final_theta"], fwd["final_S"]),
        mu_state_init=(fwd["final_mu_x"], fwd["final_mu_dx"], fwd["final_count"]),
        se_mode=se_mode, se_window=se_window,
    )

    n_fill = fill_window if fill_window is not None else fwd["w"]

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in fwd["result"].items()}
    for key in ("T_t", "tau_t", "SE_t", "C_t", "dC_t"):
        if key not in out:
            continue
        bwd_reversed = bwd["result"][key][::-1]
        out[key][:n_fill] = bwd_reversed[:n_fill]

    return out


def to_xarray(result: dict, time=None, var_names: list[str] | None = None):
    """Wrap MtvLK output in an xarray Dataset (requires xarray).

    Parameters
    ----------
    result     : output dict from compute_mtvlk
    time       : optional time coordinate array of length N-1
    var_names  : optional list of n variable names

    Returns
    -------
    xarray.Dataset
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray is required: pip install mtvlk[xarray]") from e

    T_t = result["T_t"]
    tau_t = result["tau_t"]
    n = T_t.shape[1]
    vnames = var_names or [f"x{i}" for i in range(n)]
    t_coord = time if time is not None else result["time_idx"]

    coords = {"time": t_coord, "source": vnames, "target": vnames}
    ds = xr.Dataset(
        {
            "T": xr.DataArray(T_t, dims=["time", "source", "target"], coords=coords,
                              attrs={"units": "nats/time", "long_name": "absolute LK information flow"}),
            "tau": xr.DataArray(tau_t, dims=["time", "source", "target"], coords=coords,
                                attrs={"units": "%", "long_name": "relative LK information flow"}),
        }
    )
    if "C_t" in result:
        ds["C"] = xr.DataArray(result["C_t"], dims=["time", "source", "target"], coords=coords)
        ds["dC"] = xr.DataArray(result["dC_t"], dims=["time", "source", "target"], coords=coords)
    return ds
