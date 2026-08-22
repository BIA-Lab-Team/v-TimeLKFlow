# v-TimeLKFlow

**Time-varying Liang–Kleeman information flow implementations** — a
BIA Lab package. Its current main implementation (`mtvlk`) is Zhou et al.
(2024)'s Kalman-filter-based method for estimating *time-varying,
multivariate* causal information flow between time series.

License: MIT

## Contents

- [Time-varying vs. static causality](#time-varying-vs-static-causality)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Package contents](#package-contents)
- [API overview](#api-overview)
- [Tutorials](#tutorials)
- [On methodological reconstructions](#on-methodological-reconstructions)
- [Testing](#testing)
- [References](#references)
- [Contributing / Issues](#contributing--issues)
- [License](#license)
- [Citation](#citation)

## Time-varying vs. static causality

Most causality analyses — classical Granger causality, and the original
(static) Liang–Kleeman information flow itself — collapse an entire
observation record into a **single number per variable pair**: "does `X`
cause `Y`?" is answered once, from the whole time series, and that one
verdict is assumed to hold for the entire record. That's a poor fit for
systems where a causal relationship switches on, strengthens, weakens, or
disappears partway through — a coupling that only exists during part of a
climate regime, for example, can average out to "no detectable causality"
under a whole-series estimate even though it was clearly present for a
while. Instead of one static answer, this package tracks a **running,
time-indexed estimate**: a Kalman filter updates the local covariance
structure as each new observation arrives, so the causal-flow strength
`T_{j→i}(t)` (and its significance) is itself a function of time `t`, not a
single scalar — you get a curve showing when a link turned on or off, not
just whether it exists anywhere in the record.

**Input format**: every function in this package takes a single array `X`
of shape `(T, n)` — `T` time points (rows) by `n` signals/variables
(columns), i.e. `n` co-observed signals sampled at the same `T` time steps.
Passing all `n` signals into one call makes every estimated flow
*conditional on the other `n − 2` variables* automatically — unlike a purely
bivariate/pairwise method, which can report a spurious link that's actually
mediated entirely through an unmodeled third signal (see the worked
bivariate-vs-conditional example in
[`examples/fig2_fig3_replication.py`](examples/fig2_fig3_replication.py)).

## Overview

The Liang–Kleeman (LK) information flow (Liang 2021) is a rigorous,
first-principles measure of causality between time series, computed directly
from sample covariances — no model fitting, no surrogate data, and (unlike
Granger causality) a provable "principle of nil causality": if `X` truly
doesn't cause `Y`, the estimated flow is exactly zero in the population
limit.

The classical LK formalism assumes the underlying covariance structure is
**stationary** — a single number for the whole time series. Hagan et al.
(2019) extended this to a **time-varying bivariate** form (`TvLK`) by
tracking the covariances with a Kalman filter instead of a single sample
estimate. Zhou et al. (2024) generalized that further to the **multivariate**
case (`MtvLK`), so you can ask "does `X1` cause `X2`, *conditional on*
`X3, ..., Xn`?" and have the answer change over time.

This package implements the full multivariate, time-varying method, plus the
paper's Kalman-filter noise-estimation and significance-testing machinery —
including places where the paper's own description is underspecified, which
are documented explicitly rather than silently guessed at (see
[On methodological reconstructions](#on-methodological-reconstructions)).

- Zhou, F., et al. (2024). Estimating time-dependent structures in a
  multivariate causality for land–atmosphere interactions. *Journal of
  Climate*, 37(6). [10.1175/JCLI-D-23-0207.1](https://doi.org/10.1175/JCLI-D-23-0207.1)
- Hagan, D. F. T., et al. (2019). A time-varying causality formalism based on
  the Liang–Kleeman information flow for analyzing directed interactions in
  nonstationary climate systems. *Journal of Climate*, 32(21).
  [10.1175/JCLI-D-18-0881.1](https://doi.org/10.1175/JCLI-D-18-0881.1)
- Liang, X. S. (2021). Normalized multivariate time series causality analysis
  and causal graph reconstruction. *Entropy*, 23(6), 679.
  [10.3390/e23060679](https://doi.org/10.3390/e23060679)

## Features

- **Static multivariate LK** (`compute_lk`) — the stationary baseline, with
  block-bootstrap standard errors.
- **Time-varying MtvLK** (`compute_mtvlk`) — a Square-Root Kalman Filter
  tracks the covariance structure through time, giving `T(t)` instead of a
  single `T`.
- **Adaptive process/measurement noise** (`qr_mode="adaptive"`) — the
  paper's Kalman filter noise covariances `Q`/`R` aren't held constant; this
  package estimates them online instead of requiring you to hand-tune them.
- **Appendix A window-fill** (`compute_mtvlk_filled`) — the Kalman filter
  needs a lookback window to "warm up," during which its estimate is
  unreliable. This runs a second pass backward through time to fill that
  initial gap, following the paper's own Appendix A prescription.
- **Three significance-test modes** (`se_mode="kf"|"classical"|"regression"`)
  — from the filter's own uncertainty tracking, a closed-form classical
  approximation, or a literal windowed-regression implementation of the
  paper's stated Fisher-information formula. See each mode's docstring in
  [`mtvlk/core/mtvlk.py`](mtvlk/core/mtvlk.py) for when to reach for which.
- **A full synthetic-model replication suite** ([`examples/`](examples/)) —
  every synthetic experiment in Zhou et al. (2024) (Figures 2–4 and the
  Appendix C null-link tests), runnable as scripts or notebooks. See
  [Tutorials](#tutorials).

## Installation

The PyPI distribution is named `v-TimeLKFlow` (the lab's umbrella name for this
family of time-varying LK flow implementations); the importable Python package
remains `mtvlk` (the specific Zhou et al. 2024 algorithm this release implements).

```bash
pip install v-TimeLKFlow
```

Optional extras:

```bash
pip install v-TimeLKFlow[viz]        # + matplotlib, for mtvlk.viz plotting helpers
pip install v-TimeLKFlow[xarray]     # + xarray, for to_xarray() output
pip install v-TimeLKFlow[all]        # viz + xarray + numba
pip install v-TimeLKFlow[dev]        # + pytest, for running the test suite
pip install v-TimeLKFlow[notebooks]  # + jupyter, for the tutorial notebooks
```

Core dependencies are just `numpy` and `scipy` — everything else is opt-in.

## Quick start

```python
import numpy as np
from mtvlk import compute_lk, compute_mtvlk, compute_mtvlk_filled

rng = np.random.default_rng(0)
X = rng.standard_normal((500, 3))  # 500 time steps, 3 variables

# Static multivariate LK (Liang 2021) -- one T matrix for the whole series
result_static = compute_lk(X, dt=1.0)
print(result_static["T"])            # (3, 3) causal matrix, T[j, i] = T_{j->i}

# Time-varying MtvLK (Zhou et al. 2024) -- a T(t) for every time step
result_tv = compute_mtvlk(X, dt=1.0)
print(result_tv["T_t"].shape)        # (499, 3, 3)

# With the paper's adaptive noise + Appendix A window-fill + a real
# significance test, all at once:
result = compute_mtvlk_filled(
    X, dt=1.0,
    qr_mode="adaptive", qr_window=100,
    fill_window=100,
    se_mode="classical", se_window=100,
    return_SE=True,
)
print(result["T_t"].shape, result["SE_t"].shape)
```

## Package contents

| Path | Contents |
|---|---|
| [`mtvlk/core/lk_stationary.py`](mtvlk/core/lk_stationary.py) | Static multivariate LK (`compute_lk`), the core cofactor-matrix formula, block-bootstrap standard errors. |
| [`mtvlk/core/sqkf.py`](mtvlk/core/sqkf.py) | `SquareRootKF` — the Cholesky square-root Kalman filter that tracks the packed covariance/tendency-covariance state, plus the closed-form `analytical_R` noise formula. |
| [`mtvlk/core/mtvlk.py`](mtvlk/core/mtvlk.py) | `compute_mtvlk` / `compute_mtvlk_filled` / `to_xarray` — the time-varying MtvLK driver, adaptive-noise wiring, and the Appendix A backward/forward splice. |
| [`mtvlk/utils/adaptive_noise.py`](mtvlk/utils/adaptive_noise.py) | `AdaptiveQR` — online EWMA estimator of the Kalman filter's process noise `Q_t`. |
| [`mtvlk/utils/regression_se.py`](mtvlk/utils/regression_se.py) | `RegressionSE` — windowed VAR(1) regression-coefficient variance, a literal implementation of Hagan (2019) appendix C's significance formula. |
| [`mtvlk/utils/covariance.py`](mtvlk/utils/covariance.py) | Low-level helpers: finite differences, sample/tendency covariance, cofactor matrices, upper-triangular packing. |
| [`mtvlk/utils/significance.py`](mtvlk/utils/significance.py) | `pointwise_ttest` — an alternative block-permutation significance test. |
| [`mtvlk/viz/plots.py`](mtvlk/viz/plots.py) | Optional plotting helpers (causal-matrix heatmaps, time-series plots, static-vs-time-varying comparisons) — requires `mtvlk[viz]`. |
| [`tests/`](tests/) | The pytest suite — one test file per `mtvlk/` module, plus end-to-end checks in `test_mtvlk.py`. |
| [`examples/`](examples/) | The full synthetic-model replication suite (see [Tutorials](#tutorials) below). |
| [`papers/`](papers/) | The reference paper PDF, kept for convenience — **not** shipped in the PyPI package. |

## API overview

- **`compute_lk(X, dt, n_boot=200, block_len=None, rng=None)`** — static
  multivariate LK over the whole series. Returns `T`, `tau`, and bootstrap
  standard errors `T_err`/`tau_err`.
- **`compute_mtvlk(X, dt, ...)`** — time-varying MtvLK. Key parameters:
  - `qr_mode`: `"constant"` (fixed `Q_scale`/`R_scale`, the paper's naive
    baseline) or `"adaptive"` (online noise estimation).
  - `init_window`: how many initial samples seed the Kalman filter (the
    paper uses 300 for its 3-variable models, 200 for its 5-variable one).
  - `se_mode`: `"kf"`, `"classical"`, or `"regression"` — see each mode's
    docstring for the trade-offs.
  - `return_covariances` / `return_SE`: also return the tracked `C_t`/`dC_t`
    and/or the standard error `SE_t`.
- **`compute_mtvlk_filled(X, dt, ..., fill_window=None)`** — same as
  `compute_mtvlk`, plus the Appendix A backward-pass splice that fills in the
  initial `init_window`-length gap. This is what the example scripts use.
- **`SquareRootKF`** — the underlying filter class, if you need lower-level
  access (e.g. to `analytical_R` directly).
- **`to_xarray(result, time=None, var_names=None)`** — wrap a
  `compute_mtvlk`/`compute_mtvlk_filled` result as an `xarray.Dataset`
  (`mtvlk[xarray]` required).

Every parameter is documented in its function's docstring — read those for
the full picture; this section is a map, not a duplicate.

## Tutorials

The [`examples/`](examples/) directory replicates every synthetic experiment
in Zhou et al. (2024), as both standalone scripts and Jupyter notebooks. See
[`examples/README.md`](examples/README.md) for a full walkthrough of each
one, what it demonstrates, and its expected output. In brief:

| Script | Notebook | Demonstrates |
|---|---|---|
| [`fig2_fig3_replication.py`](examples/fig2_fig3_replication.py) | [`01_fig2_fig3_replication.ipynb`](examples/notebooks/01_fig2_fig3_replication.ipynb) | Synthetic Models 1 & 2 (paper Figs. 2–3): a 3-variable causal network whose links switch on and back off, bivariate vs. conditional causality, and a worked example of a *spurious* bivariate link that vanishes once conditioned on the confounder. |
| [`fig4_model3_replication.py`](examples/fig4_model3_replication.py) | [`02_fig4_model3_replication.ipynb`](examples/notebooks/02_fig4_model3_replication.ipynb) | Synthetic Model 3 (paper Fig. 4): a 5-variable network that switches on and *stays* on, testing multivariate conditioning with several mediating paths at once. |
| [`fig_appendixC_replication.py`](examples/fig_appendixC_replication.py) | [`03_fig_appendixC_replication.ipynb`](examples/notebooks/03_fig_appendixC_replication.ipynb) | Appendix C: every non-causal pair in all three models, confirming the method reports no significant flow where none was designed in. |
| [`synthetic_replication.py`](examples/synthetic_replication.py) | [`04_synthetic_replication.ipynb`](examples/notebooks/04_synthetic_replication.ipynb) | Supplementary validation scenarios (abrupt regime shift, a trivariate confounder, sinusoidal coupling) not tied to a specific paper figure. |

Each script/notebook takes roughly 1–5 minutes to run (they use a reduced
ensemble size compared to the paper's 1000 realizations — see each file's
header comment for the exact reduction and why it doesn't change the
qualitative conclusions).

## On methodological reconstructions

The paper describes two mechanisms without giving an exact, reproducible
formula, and its own code repository (linked in the paper) is no longer
available. This package is transparent about where it had to reconstruct
rather than copy:

- **Adaptive `Q`/`R` noise estimation** (`qr_mode="adaptive"`,
  [`AdaptiveQR`](mtvlk/utils/adaptive_noise.py)) — the paper says it uses an
  "exponentially weighted moving average" but doesn't give the update
  equation. This package's reconstruction is documented in
  `adaptive_noise.py`'s module docstring, including the specific alternative
  formulations that were tried and rejected during development (an earlier
  design caused a destabilizing feedback loop; see the docstring for why).
- **`se_mode="classical"`** — a closed-form approximation to the paper's
  significance test, decoupled from the Kalman filter's own `Q`/`R`. Verified
  to be within the right order of magnitude of the paper's own stated
  formula, but not an exact algebraic match (see
  `mtvlk/utils/regression_se.py`'s module docstring for the ~20% gap and why
  it's expected).
- **`se_mode="regression"`** — the most literal reconstruction, directly
  implementing the windowed VAR(1)-regression Fisher-information formula
  that Hagan et al. (2019) Appendix C states explicitly. At `n=2` (the
  bivariate case both papers describe most precisely), this has been checked
  against a from-scratch OLS calculation in `tests/test_mtvlk.py`.

Where the underlying causality formula itself is concerned (the actual `T`
computation, `_lk_from_moments`), there is no ambiguity — it's a direct,
verified implementation of Liang (2021)'s published formula, confirmed this
project to reduce to Hagan et al. (2019)'s bivariate formula to machine
precision at `n=2`.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## References

- Liang, X. S. (2021). Normalized multivariate time series causality
  analysis and causal graph reconstruction. *Entropy*, 23(6), 679.
  [10.3390/e23060679](https://doi.org/10.3390/e23060679)
- Hagan, D. F. T., Wang, G., Liang, X. S., & Dolman, H. A. J. (2019). A
  time-varying causality formalism based on the Liang–Kleeman information
  flow for analyzing directed interactions in nonstationary climate
  systems. *Journal of Climate*, 32(21).
  [10.1175/JCLI-D-18-0881.1](https://doi.org/10.1175/JCLI-D-18-0881.1)
- Zhou, F., Hagan, D. F. T., Wang, G., Liang, X. S., Li, S., Shao, Y.,
  Yeboah, E., & Wei, X. (2024). Estimating time-dependent structures in a
  multivariate causality for land–atmosphere interactions. *Journal of
  Climate*, 37(6).
  [10.1175/JCLI-D-23-0207.1](https://doi.org/10.1175/JCLI-D-23-0207.1)

## Contributing / Issues

Bug reports, questions, and pull requests are welcome — please open an issue
at the [repository's issue tracker](https://github.com/BIA-Lab-Team/v-TimeLKFlow/issues).

## License

[MIT](LICENSE) © 2026 BIA Lab

## Citation

There is no standalone paper for this software (yet) — please cite the GitHub
repository for the software itself, and the paper for the algorithm it
implements:

```bibtex
@misc{v_timelkflow,
  author       = {Zhou, Felix Y.},
  title        = {v-TimeLKFlow: Time-Varying Liang-Kleeman Information Flow Implementations},
  year         = {2026},
  howpublished = {\url{https://github.com/BIA-Lab-Team/v-TimeLKFlow}},
  license      = {MIT}
}

@article{zhou2024mtvlk,
  author  = {Zhou, Feihong and Hagan, Daniel Fiifi Tawia and Wang, Guojie and
             Liang, X. San and Li, Shijie and Shao, Yuhao and Yeboah, Emmanuel
             and Wei, Xikun},
  title   = {Estimating Time-Dependent Structures in a Multivariate Causality
             for Land--Atmosphere Interactions},
  journal = {Journal of Climate},
  volume  = {37},
  number  = {6},
  year    = {2024},
  doi     = {10.1175/JCLI-D-23-0207.1}
}
```
