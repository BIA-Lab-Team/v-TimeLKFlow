"""Replication of Figure 4 from Zhou et al. (2024) -- Synthetic Model 3 (Eq. 19).

A 5-variable AR(1) system: x1 is autonomous; x2<-x1; x3<-x2; x4<-x2,x3;
x5<-x1,x4. All six coupling coefficients share one ramp schedule: 0 for
t<=500, linearly increasing 0.0005/step for 500<t<=1500 (reaching 0.5),
then HELD at 0.5 for 1500<t<=2000 -- unlike synthetic models 1/2, this
network turns on and stays on (no ramp-down).

Reference: Zhou et al. (2024). J. Climate 37(6), doi:10.1175/JCLI-D-23-0207.1
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

from mtvlk import compute_mtvlk_filled

# ── Parameters ──────────────────────────────────────────────────────────────
N = 2000          # time series length (paper: 2000)
N_REAL = 30       # realizations (paper: 1000; 30 gives reasonable statistics)
DT = 1.0
Q_SCALE = 5e-5    # initial/fallback process noise for the adaptive estimator
R_SCALE = 1.0     # initial/fallback observation noise for the adaptive estimator
INIT_WIN = 200    # lookback window (paper states 200 for this experiment,
                  # vs. 300 for synthetic models 1/2)
QR_MODE = "adaptive"
QR_WINDOW = INIT_WIN
# Splice window for the Appendix A backward-pass fill -- see the discussion
# in fig2_fig3_replication.py for why this needs to balance two competing
# artifacts. 1.5x the lookback window matched the sweet spot found there
# (300/450 for the 3-variable, window=300 case); re-verified empirically for
# this 5-variable, window=200 case below rather than assumed.
FILL_WINDOW = 300
SE_MODE = "classical"   # Hagan et al. (2019) appendix C / Zhou et al. (2024) appendix B
SE_WINDOW = INIT_WIN
Z_99 = 2.326
SMOOTH = 50

# Ramp schedule (1-indexed as in paper): 0 for t<=500, 0.0005/step rise for
# 500<t<=1500 (reaching 0.5 exactly at t=1500), held at 0.5 for t>1500.
_T_AX = np.arange(1, N + 1)
_C_FULL = np.full(N, 0.5)
_C_FULL[_T_AX <= 500] = 0.0
mask_ramp = (_T_AX > 500) & (_T_AX <= 1500)
_C_FULL[mask_ramp] = 0.0005 * (_T_AX[mask_ramp] - 500)

# KEYS: the 6 true links, each a (source_idx, target_idx) pair into the full
# 5x5 T_t/SE_t arrays (0-indexed: x1=0, x2=1, x3=2, x4=3, x5=4).
KEYS = {
    "c12_345": (0, 1),  # x1->x2 | x3,x4,x5
    "c15_234": (0, 4),  # x1->x5 | x2,x3,x4
    "c23_145": (1, 2),  # x2->x3 | x1,x4,x5
    "c24_135": (1, 3),  # x2->x4 | x1,x3,x5
    "c34_125": (2, 3),  # x3->x4 | x1,x2,x5
    "c45_123": (3, 4),  # x4->x5 | x1,x2,x3
}
LABELS = {
    "c12_345": r"|T$_{x1\to x2|x3,x4,x5}$|",
    "c15_234": r"|T$_{x1\to x5|x2,x3,x4}$|",
    "c23_145": r"|T$_{x2\to x3|x1,x4,x5}$|",
    "c24_135": r"|T$_{x2\to x4|x1,x3,x5}$|",
    "c34_125": r"|T$_{x3\to x4|x1,x2,x5}$|",
    "c45_123": r"|T$_{x4\to x5|x1,x2,x3}$|",
}
PANEL_LETTERS = {
    "c12_345": "c", "c15_234": "d", "c23_145": "e",
    "c24_135": "f", "c34_125": "g", "c45_123": "h",
}


# ── Synthetic model simulator ────────────────────────────────────────────────

def sim_model3(rng):
    """Eq. (19): x1->x2->x3->x4<-x2, x4->x5<-x1."""
    X = np.zeros((N, 5))
    for t in range(1, N):
        c = _C_FULL[t - 1]
        eps = rng.standard_normal(5)
        X[t, 0] = 0.35 * X[t-1, 0] + eps[0]
        X[t, 1] = 0.35 * X[t-1, 1] + c * X[t-1, 0] + eps[1]
        X[t, 2] = 0.35 * X[t-1, 2] + c * X[t-1, 1] + eps[2]
        X[t, 3] = 0.35 * X[t-1, 3] + c * X[t-1, 1] + c * X[t-1, 2] + eps[3]
        X[t, 4] = 0.35 * X[t-1, 4] + c * X[t-1, 0] + c * X[t-1, 3] + eps[4]
    return X


# ── Ensemble runner ──────────────────────────────────────────────────────────

def run_ensemble(verbose=True):
    T_len = N - 1
    store = {k: np.zeros((N_REAL, T_len)) for k in KEYS}
    store.update({"se_" + k: np.zeros((N_REAL, T_len)) for k in KEYS})

    t0 = time.time()
    for r in range(N_REAL):
        rng = np.random.default_rng(r)
        X = sim_model3(rng)

        res = compute_mtvlk_filled(X, dt=DT, Q_scale=Q_SCALE, R_scale=R_SCALE,
                                    init_window=INIT_WIN, qr_mode=QR_MODE,
                                    qr_window=QR_WINDOW, fill_window=FILL_WINDOW,
                                    se_mode=SE_MODE, se_window=SE_WINDOW,
                                    return_SE=True)
        T_t = res["T_t"]
        SE_t = res["SE_t"]
        for key, (j, i) in KEYS.items():
            store[key][r] = np.abs(T_t[:, j, i])
            store["se_" + key][r] = SE_t[:, j, i]

        if verbose and (r + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (r + 1) * (N_REAL - r - 1)
            print(f"  r={r+1:3d}/{N_REAL}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return store


def sig_threshold(store):
    """Time-varying 1% significance for the ensemble-mean |T| curve --
    z_99 * mean(SE) / sqrt(N_REAL), since the plotted curve is a mean over
    N_REAL independent realizations (see fig2_fig3_replication.py's
    sig_threshold docstring for why the sqrt(N_REAL) division matters)."""
    return {k: Z_99 * store["se_" + k].mean(axis=0) / np.sqrt(N_REAL) for k in KEYS}


# ── 5-node causal diagram helper ─────────────────────────────────────────────

def _draw_node(ax, xy, label, color="lightyellow"):
    box = mpatches.FancyBboxPatch(
        (xy[0] - 0.09, xy[1] - 0.09), 0.18, 0.18,
        boxstyle="round,pad=0.02", fc=color, ec="black", lw=1.2, zorder=3
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=8, fontweight="bold", zorder=4)


def _draw_arrow(ax, src, tgt, color="red"):
    dx, dy = tgt[0] - src[0], tgt[1] - src[1]
    norm = (dx**2 + dy**2)**0.5
    shrink = 0.13 * norm * 80
    arrow = FancyArrowPatch(
        src, tgt, arrowstyle="Simple,tail_width=1.3,head_width=7,head_length=5",
        color=color, shrinkA=shrink, shrinkB=shrink, zorder=2
    )
    ax.add_patch(arrow)


# Pentagon layout matching the paper: x2 top, x3 upper-right, x4 lower-right,
# x5 lower-left, x1 upper-left.
_NODE_POS = [
    (0.120, 0.624),  # x1
    (0.500, 0.900),  # x2
    (0.880, 0.624),  # x3
    (0.735, 0.176),  # x4
    (0.265, 0.176),  # x5
]
_NODE_LABELS = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
_ALL_LINKS = [(0, 1), (1, 2), (1, 3), (2, 3), (0, 4), (3, 4)]


def causal_diagram_ax(ax, links, title=""):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=2)
    for src_i, tgt_i in links:
        _draw_arrow(ax, _NODE_POS[src_i], _NODE_POS[tgt_i])
    for xy, lbl in zip(_NODE_POS, _NODE_LABELS):
        _draw_node(ax, xy, lbl)


# ── Panel plot helper ────────────────────────────────────────────────────────

def _smooth(arr, w):
    """Rolling-mean smooth with edge padding to avoid edge effects (plain
    convolve(mode="same") implicitly zero-pads, biasing the curve toward
    zero near the edges -- visible here since this model's signal stays
    high at the right edge, unlike models 1/2 which ramp back to ~0)."""
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:pad + len(arr)]


def panel_plot(ax, t_ax, mean_arr, sig_arr, label, panel_letter, ylim=0.25):
    mean_s = _smooth(mean_arr, SMOOTH)
    sig_s = _smooth(sig_arr, SMOOTH)

    ax.fill_between(t_ax, 0, mean_s, alpha=0.20, color="#2166AC")
    ax.plot(t_ax, mean_s, color="#2166AC", lw=1.5, label=label)
    ax.plot(t_ax, sig_s, color="#D73027", lw=1.2, ls="-", label="Sig (1% level)")

    ax.axvline(500, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.axvline(1500, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, N - 1)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("|T| (nats/step)", fontsize=8)
    ax.set_title(f"({panel_letter}) {label}", fontsize=9, pad=3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", handlelength=1.2)


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def make_figure4(store, sig):
    t_ax = np.arange(N - 1)

    fig = plt.figure(figsize=(11, 12))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.09, right=0.97, top=0.92, bottom=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500\n(no causal links)")
    causal_diagram_ax(ax_b, links=_ALL_LINKS,
                       title="(b) 500<t≤2000\n(x1→x2→x3→x4←x2, x1→x5←x4)")

    for idx, key in enumerate(["c12_345", "c15_234", "c23_145",
                                "c24_135", "c34_125", "c45_123"]):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        panel_plot(ax, t_ax, store[key].mean(0), sig[key],
                   LABELS[key], PANEL_LETTERS[key], ylim=0.25)

    fig.suptitle(
        f"Figure 4 — Synthetic Model 3 (Eq. 19)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations  ·  "
        f"red = 1% significance (Fisher-information SE)",
        fontsize=10, fontweight="bold", y=0.995
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    print(f"Running Synthetic Model 3 ensemble (N_REAL={N_REAL}, N={N})...")
    store = run_ensemble()
    sig = sig_threshold(store)

    print("\nGenerating Figure 4...")
    fig4 = make_figure4(store, sig)
    path4 = os.path.join(out_dir, "fig4_model3_replication.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    print("  Saved: " + path4)
    plt.close(fig4)

    # ── Quick verification ────────────────────────────────────────────────────
    t_ax = np.arange(N - 1)
    null_period = t_ax <= 500
    coupling_period = t_ax > 1500   # fully-saturated plateau region

    def check(name, arr, expect_sig=True):
        mean_t = arr.mean(0)
        peak = mean_t[coupling_period].mean()
        quiet = mean_t[null_period].mean()
        ratio = peak / (quiet + 1e-12)
        marker = "OK" if (ratio > 2) == expect_sig else "XX"
        print(f"  {marker} {name:35s}  peak/quiet ratio = {ratio:.2f}  peak_val = {mean_t.max():.4f}")

    print("\nVerification (coupling-plateau vs null period ratio):")
    for key in ["c12_345", "c15_234", "c23_145", "c24_135", "c34_125", "c45_123"]:
        check(f"{key} {LABELS[key]}", store[key], True)

    print("\nSig line peak values:")
    for key in KEYS:
        print(f"  {key:12s}  sig_peak = {sig[key].max():.4f}")

    print("\nDone.")
