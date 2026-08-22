"""mtvlk — Multivariate Time-Varying Liang-Kleeman Information Flow.

Reference: Zhou et al. (2024). Estimating time-dependent structures in a
multivariate causality for land-atmosphere interactions.
Journal of Climate, 37(6). DOI: 10.1175/JCLI-D-23-0207.1

Quick start
-----------
>>> import numpy as np
>>> from mtvlk import compute_lk, compute_mtvlk
>>>
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((500, 3))  # 500 time steps, 3 variables
>>>
>>> # Static multivariate LK (Liang 2021)
>>> result_static = compute_lk(X, dt=1.0)
>>> print(result_static["T"])          # (3, 3) causal matrix
>>>
>>> # Time-varying MtvLK (Zhou et al. 2024)
>>> result_tv = compute_mtvlk(X, dt=1.0)
>>> print(result_tv["T_t"].shape)      # (499, 3, 3)
"""

from mtvlk.core.lk_stationary import compute_lk
from mtvlk.core.mtvlk import compute_mtvlk, compute_mtvlk_filled, to_xarray
from mtvlk.core.sqkf import SquareRootKF

__all__ = [
    "compute_lk",
    "compute_mtvlk",
    "compute_mtvlk_filled",
    "to_xarray",
    "SquareRootKF",
]
