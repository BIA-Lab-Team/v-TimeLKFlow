"""Plotting utilities for LK information flow results."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError as e:
    raise ImportError("matplotlib is required: pip install mtvlk[viz]") from e


def plot_causal_matrix(
    T: np.ndarray,
    tau: np.ndarray | None = None,
    var_names: list[str] | None = None,
    T_err: np.ndarray | None = None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str = "LK Information Flow",
    use_tau: bool = False,
) -> plt.Axes:
    """Heatmap of the static causal matrix T or tau.

    Parameters
    ----------
    T        : (n, n) absolute flow matrix  [T[j,i] = T_{j->i}]
    tau      : (n, n) relative flow [%], plotted if use_tau=True
    var_names: list of n variable labels (columns = targets, rows = sources)
    T_err    : (n, n) standard errors — overlaid as text
    ax       : existing matplotlib Axes; created if None
    cmap     : colormap name (diverging recommended)
    title    : plot title
    use_tau  : if True, plot tau instead of T
    """
    data = tau if (use_tau and tau is not None) else T
    n = data.shape[0]
    labels = var_names or [f"x{i}" for i in range(n)]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, n), max(3, n)))

    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data.T, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")
    ax.set_title(title)

    # Annotate each cell
    for j in range(n):
        for i in range(n):
            val = data[j, i]
            text = f"{val:.2f}"
            if T_err is not None:
                text += f"\n±{T_err[j,i]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 * vmax else "black")

    plt.colorbar(im, ax=ax, label="% relative flow" if use_tau else "nats/time")
    return ax


def plot_causal_timeseries(
    T_t: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
    time: np.ndarray | None = None,
    var_names: list[str] | None = None,
    tau_t: np.ndarray | None = None,
    use_tau: bool = False,
    ax=None,
    title: str = "Time-Varying LK Information Flow",
) -> plt.Axes:
    """Line plot of T_{j->i}(t) for selected causal pairs.

    Parameters
    ----------
    T_t      : (N-1, n, n) time-varying absolute flow
    pairs    : list of (j, i) source→target pairs to plot; defaults to all off-diagonal
    time     : (N-1,) time coordinate; uses integer index if None
    var_names: list of n variable labels
    tau_t    : (N-1, n, n) relative flow — used if use_tau=True
    use_tau  : plot tau instead of T
    ax       : existing Axes; created if None
    title    : plot title
    """
    data = tau_t if (use_tau and tau_t is not None) else T_t
    N_steps, n, _ = data.shape
    labels = var_names or [f"x{i}" for i in range(n)]
    t = time if time is not None else np.arange(N_steps)

    if pairs is None:
        pairs = [(j, i) for j in range(n) for i in range(n) if j != i]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for j, i in pairs:
        ax.plot(t, data[:, j, i], label=f"T({labels[j]}→{labels[i]})")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("% relative flow" if use_tau else "nats/time")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_comparison(
    T_static: np.ndarray,
    T_t: np.ndarray,
    pair: tuple[int, int],
    time: np.ndarray | None = None,
    var_names: list[str] | None = None,
    ax=None,
) -> plt.Axes:
    """Compare static LK and time-varying MtvLK for a single pair (j→i).

    Parameters
    ----------
    T_static : (n, n) static T
    T_t      : (N-1, n, n) time-varying T
    pair     : (j, i) source→target indices
    """
    j, i = pair
    n = T_static.shape[0]
    labels = var_names or [f"x{i_}" for i_ in range(n)]
    t = time if time is not None else np.arange(T_t.shape[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, T_t[:, j, i], label="MtvLK (time-varying)", color="steelblue")
    ax.axhline(T_static[j, i], color="tomato", linestyle="--",
               label=f"Static LK = {T_static[j, i]:.3f}")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time")
    ax.set_ylabel("T (nats/time)")
    ax.set_title(f"T({labels[j]}→{labels[i]}): Static vs Time-Varying")
    ax.legend()
    return ax
