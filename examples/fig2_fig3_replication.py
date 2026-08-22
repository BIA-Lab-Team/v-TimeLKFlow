"""Replication of Figures 2 and 3 from Zhou et al. (2024).

Synthetic Model 1 (Fig 2): 3-var AR(1) with bidirectional x1<->x2 and x2->x3.
Synthetic Model 2 (Fig 3): 3-var AR(1) with x2->x1, x2->x3, x1->x3 (x2 autonomous).

Both use a triangular coupling schedule: ramp 0->0.5 for 500<t<=1000,
ramp 0.5->0 for 1000<t<=1500, zero elsewhere.

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

# ── Parameters ────────────────────────────────────────────────────────────────
N = 2000          # time series length (paper: 2000)
N_REAL = 30       # realizations (paper: 1000; 30 gives reasonable statistics)
DT = 1.0
Q_SCALE = 5e-5    # initial/fallback process noise for the adaptive estimator
R_SCALE = 1.0     # initial/fallback observation noise for the adaptive estimator
INIT_WIN = 300    # warm-up window length (paper: 300)
QR_MODE = "adaptive"   # paper §2b: Q/R estimated via EWMA/UWMA, not held constant
QR_WINDOW = INIT_WIN   # paper uses a single window-length choice (300) for both
# Splice window for the Appendix A backward-pass fill. There are two
# competing artifacts as this widens: (1) below ~350-450, the forward pass's
# adaptive-estimator "hangover" (its rolling window still contains noisy
# replayed warm-up samples) shows through as a small bump right after
# init_window; (2) above ~450-500, the backward pass's own KF lag as it
# unwinds through the coupling ramp in reverse shows through as a bump
# just before the true ramp starts, growing quickly with window size.
# 450 empirically minimizes both (swept 300-600 on both synthetic models).
# The paper's own appendix A already cautions that the spliced region is a
# numerical patch, not a physically faithful reconstruction, so this is a
# practical minimum, not a claim of exactly matching the paper's choice.
FILL_WINDOW = 450
# Hagan et al. (2019) appendix C / Zhou et al. (2024) appendix B: the
# significance test is a Fisher-information/MLE variance of a local AR-model
# fit over a window of raw data -- decoupled from the Kalman filter's own Q/R
# -- not derived from the filter's own state covariance P. "classical" uses
# mtvlk's reconstruction of that (see mtvlk.core.mtvlk's se_mode docstring).
SE_MODE = "classical"
SE_WINDOW = INIT_WIN   # paper's single window-length choice (300)
Z_99 = 2.326      # one-sided z-score for 1% significance level
SMOOTH = 50       # rolling-mean window for display

# Coupling ramp values (1-indexed as in paper)
_T_AX = np.arange(1, N + 1)
_C_FULL = np.zeros(N)
mask_up   = (_T_AX > 500) & (_T_AX <= 1000)
mask_down = (_T_AX > 1000) & (_T_AX <= 1500)
_C_FULL[mask_up]   = 0.001 * (_T_AX[mask_up] - 500)
_C_FULL[mask_down] = 0.001 * (1500 - _T_AX[mask_down])

KEYS = ["biv_12", "biv_21", "biv_23", "biv_13",
        "tri_12", "tri_21", "tri_23", "tri_13"]


# ── Synthetic model simulators ─────────────────────────────────────────────────

def sim_model1(rng):
    """Eq. (17): bidirectional x1<->x2, x2->x3 coupling."""
    X = np.zeros((N, 3))
    for t in range(1, N):
        c = _C_FULL[t - 1]
        eps = rng.standard_normal(3)
        X[t, 0] = 0.35 * X[t-1, 0] + c * X[t-1, 1] + eps[0]
        X[t, 1] = 0.35 * X[t-1, 1] + c * X[t-1, 0] + eps[1]
        X[t, 2] = 0.35 * X[t-1, 2] + c * X[t-1, 1] + eps[2]
    return X


def sim_model2(rng):
    """Eq. (18): x2->x1, x2->x3, x1->x3 (x2 autonomous)."""
    X = np.zeros((N, 3))
    for t in range(1, N):
        c = _C_FULL[t - 1]
        eps = rng.standard_normal(3)
        X[t, 0] = 0.35 * X[t-1, 0] + c * X[t-1, 1] + eps[0]
        X[t, 1] = 0.35 * X[t-1, 1] + eps[1]
        X[t, 2] = 0.35 * X[t-1, 2] + c * X[t-1, 0] + c * X[t-1, 1] + eps[2]
    return X


# ── Ensemble runner ────────────────────────────────────────────────────────────

def run_ensemble(sim_fn, verbose=True):
    """
    Returns a dict of shape-(N_REAL, N-1) arrays for |T| and SE for each pair.

    Fig 2 / Fig 3 pairs (0-indexed variables: x1=0, x2=1, x3=2):

      biv_12  bivariate T_{x1->x2}   from compute_mtvlk(X[:,[0,1]])[:,0,1]
      biv_21  bivariate T_{x2->x1}   from compute_mtvlk(X[:,[0,1]])[:,1,0]
      biv_23  bivariate T_{x2->x3}   from compute_mtvlk(X[:,[1,2]])[:,0,1]
      biv_13  bivariate T_{x1->x3}   from compute_mtvlk(X[:,[0,2]])[:,0,1]
      tri_12  trivariate T_{x1->x2|x3}   from compute_mtvlk(X)[:,0,1]
      tri_21  trivariate T_{x2->x1|x3}   from compute_mtvlk(X)[:,1,0]
      tri_23  trivariate T_{x2->x3|x1}   from compute_mtvlk(X)[:,1,2]
      tri_13  trivariate T_{x1->x3|x2}   from compute_mtvlk(X)[:,0,2]

    Also stores se_<key> for each key (SE of T at each time step).
    """
    T_len = N - 1
    store = {k: np.zeros((N_REAL, T_len)) for k in KEYS}
    store.update({"se_" + k: np.zeros((N_REAL, T_len)) for k in KEYS})

    t0 = time.time()
    for r in range(N_REAL):
        rng = np.random.default_rng(r)
        X = sim_fn(rng)

        # Trivariate (extracts all needed flows + SE)
        res3 = compute_mtvlk_filled(X, dt=DT, Q_scale=Q_SCALE,
                                     R_scale=R_SCALE, init_window=INIT_WIN,
                                     qr_mode=QR_MODE, qr_window=QR_WINDOW,
                                     fill_window=FILL_WINDOW, se_mode=SE_MODE, se_window=SE_WINDOW,
                                     return_SE=True)
        T3  = res3["T_t"]
        SE3 = res3["SE_t"]
        store["tri_12"][r] = np.abs(T3[:, 0, 1])
        store["tri_21"][r] = np.abs(T3[:, 1, 0])
        store["tri_23"][r] = np.abs(T3[:, 1, 2])
        store["tri_13"][r] = np.abs(T3[:, 0, 2])
        store["se_tri_12"][r] = SE3[:, 0, 1]
        store["se_tri_21"][r] = SE3[:, 1, 0]
        store["se_tri_23"][r] = SE3[:, 1, 2]
        store["se_tri_13"][r] = SE3[:, 0, 2]

        # Bivariate (x1, x2)
        res12 = compute_mtvlk_filled(X[:, [0, 1]], dt=DT, Q_scale=Q_SCALE,
                                      R_scale=R_SCALE, init_window=INIT_WIN,
                                      qr_mode=QR_MODE, qr_window=QR_WINDOW,
                                      fill_window=FILL_WINDOW, se_mode=SE_MODE, se_window=SE_WINDOW,
                                     return_SE=True)
        T12  = res12["T_t"]
        SE12 = res12["SE_t"]
        store["biv_12"][r] = np.abs(T12[:, 0, 1])
        store["biv_21"][r] = np.abs(T12[:, 1, 0])
        store["se_biv_12"][r] = SE12[:, 0, 1]
        store["se_biv_21"][r] = SE12[:, 1, 0]

        # Bivariate (x2, x3) — indices 0,1 in this 2-var sub-system map to x2,x3
        res23 = compute_mtvlk_filled(X[:, [1, 2]], dt=DT, Q_scale=Q_SCALE,
                                      R_scale=R_SCALE, init_window=INIT_WIN,
                                      qr_mode=QR_MODE, qr_window=QR_WINDOW,
                                      fill_window=FILL_WINDOW, se_mode=SE_MODE, se_window=SE_WINDOW,
                                     return_SE=True)
        store["biv_23"][r]    = np.abs(res23["T_t"][:, 0, 1])
        store["se_biv_23"][r] = res23["SE_t"][:, 0, 1]

        # Bivariate (x1, x3) — indices 0,1 map to x1,x3
        res13 = compute_mtvlk_filled(X[:, [0, 2]], dt=DT, Q_scale=Q_SCALE,
                                      R_scale=R_SCALE, init_window=INIT_WIN,
                                      qr_mode=QR_MODE, qr_window=QR_WINDOW,
                                      fill_window=FILL_WINDOW, se_mode=SE_MODE, se_window=SE_WINDOW,
                                     return_SE=True)
        store["biv_13"][r]    = np.abs(res13["T_t"][:, 0, 1])
        store["se_biv_13"][r] = res13["SE_t"][:, 0, 1]

        if verbose and (r + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (r + 1) * (N_REAL - r - 1)
            print(f"  r={r+1:3d}/{N_REAL}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return store


def sig_threshold(store):
    """Time-varying 1% significance for the ensemble-mean |T| curve.

    The blue curve plotted alongside this threshold is the MEAN of |T| over
    N_REAL realizations, whose standard error is the single-realization SE
    divided by sqrt(N_REAL) (independent realizations) -- not the raw
    single-realization SE itself. Skipping that division (an earlier bug
    here) scaled the significance line as if testing a single realization,
    making it far too large relative to the ensemble-mean curve it was
    meant to accompany -- confirmed by comparing analytical SE against the
    empirical across-realization spread of |T|, which matched reasonably
    well, showing the SE formula itself was not the problem.
    """
    thresholds = {}
    for k in KEYS:
        se_key = "se_" + k
        thresholds[k] = Z_99 * store[se_key].mean(axis=0) / np.sqrt(N_REAL)
    return thresholds


# ── Causal diagram helper ──────────────────────────────────────────────────────

def _draw_node(ax, xy, label, color="lightblue"):
    circle = mpatches.FancyBboxPatch(
        (xy[0] - 0.12, xy[1] - 0.12), 0.24, 0.24,
        boxstyle="round,pad=0.02",
        fc=color, ec="black", lw=1.2, zorder=3
    )
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=4)


def _draw_arrow(ax, src, tgt, color="red", bidirect=False):
    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    norm = (dx**2 + dy**2)**0.5
    shrink_s = 0.15 * norm
    shrink_t = 0.15 * norm
    style = "Simple,tail_width=1.5,head_width=8,head_length=6"
    arrow = FancyArrowPatch(
        src, tgt,
        arrowstyle=style,
        color=color,
        shrinkA=shrink_s * 80,
        shrinkB=shrink_t * 80,
        zorder=2
    )
    ax.add_patch(arrow)
    if bidirect:
        arrow2 = FancyArrowPatch(
            tgt, src,
            arrowstyle=style,
            color=color,
            shrinkA=shrink_s * 80,
            shrinkB=shrink_t * 80,
            zorder=2
        )
        ax.add_patch(arrow2)


def causal_diagram_ax(ax, links, title=""):
    """Draw 3-node causal diagram. links = list of (src_idx, tgt_idx, color, bidirect)."""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=2)

    # Node positions: x1 top-left, x2 top-right, x3 bottom-center
    pos = [(0.15, 0.85), (0.85, 0.85), (0.5, 0.15)]
    labels = ["$x_1$", "$x_2$", "$x_3$"]

    for lnk in links:
        src_i, tgt_i, col, bidir = lnk
        _draw_arrow(ax, pos[src_i], pos[tgt_i], color=col, bidirect=bidir)

    for i, (xy, lbl) in enumerate(zip(pos, labels)):
        _draw_node(ax, xy, lbl, color="lightyellow")


# ── Panel plot helper ──────────────────────────────────────────────────────────

def _smooth(arr, w):
    """Rolling-mean smooth with edge padding to avoid edge effects.

    convolve(..., mode="same") without pre-padding implicitly treats
    out-of-bounds samples as zero, biasing the smoothed curve toward zero
    near both edges -- invisible here only because this script's series
    happen to be near zero at both edges anyway (see
    fig4_model3_replication.py, where the signal stays high at the right
    edge and this bias showed up as a visible dip).
    """
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:pad + len(arr)]


def panel_plot(ax, t_ax, mean_arr, sig_arr, label, panel_letter, ylim=0.20):
    """Plot ensemble-mean |T| and time-varying significance line."""
    mean_s = _smooth(mean_arr, SMOOTH)
    sig_s  = _smooth(sig_arr,  SMOOTH)

    ax.fill_between(t_ax, 0, mean_s, alpha=0.20, color="#2166AC")
    ax.plot(t_ax, mean_s, color="#2166AC", lw=1.5, label=label)
    ax.plot(t_ax, sig_s,  color="#D73027", lw=1.2, ls="-", label="Sig (1% level)")

    ax.axvline(500,  color="black", lw=0.9, ls="--", alpha=0.7)
    ax.axvline(1500, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, N - 1)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("|T| (nats/step)", fontsize=8)
    ax.set_title(f"({panel_letter}) {label}", fontsize=9, pad=3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", handlelength=1.2)


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def make_figure2(store, sig):
    t_ax = np.arange(N - 1)

    fig = plt.figure(figsize=(11, 16))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.09, right=0.97, top=0.92, bottom=0.04)

    # Row 0: causal diagrams
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500 and 1500<t≤2000\n(no causal links)")
    causal_diagram_ax(ax_b, links=[
        (0, 1, "red", True),    # x1<->x2 bidirectional
        (1, 2, "red", False),   # x2->x3
    ], title="(b) 500<t≤1500\n(x1↔x2, x2→x3)")

    # Row 1: (c) biv T_{x1->x2}, (d) tri T_{x1->x2|x3}
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_plot(ax_c, t_ax, store["biv_12"].mean(0), sig["biv_12"],
               "|T$_{x1→x2}$|", "c")
    panel_plot(ax_d, t_ax, store["tri_12"].mean(0), sig["tri_12"],
               "|T$_{x1→x2|x3}$|", "d")

    # Row 2: (e) biv T_{x2->x1}, (f) tri T_{x2->x1|x3}
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    panel_plot(ax_e, t_ax, store["biv_21"].mean(0), sig["biv_21"],
               "|T$_{x2→x1}$|", "e")
    panel_plot(ax_f, t_ax, store["tri_21"].mean(0), sig["tri_21"],
               "|T$_{x2→x1|x3}$|", "f")

    # Row 3: (g) biv T_{x2->x3}, (h) tri T_{x2->x3|x1}
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])
    panel_plot(ax_g, t_ax, store["biv_23"].mean(0), sig["biv_23"],
               "|T$_{x2→x3}$|", "g")
    panel_plot(ax_h, t_ax, store["tri_23"].mean(0), sig["tri_23"],
               "|T$_{x2→x3|x1}$|", "h")

    # Row 4: (i) biv T_{x1->x3}, (j) tri T_{x1->x3|x2}
    ax_i = fig.add_subplot(gs[4, 0])
    ax_j = fig.add_subplot(gs[4, 1])
    panel_plot(ax_i, t_ax, store["biv_13"].mean(0), sig["biv_13"],
               "|T$_{x1→x3}$| (spurious)", "i")
    panel_plot(ax_j, t_ax, store["tri_13"].mean(0), sig["tri_13"],
               "|T$_{x1→x3|x2}$| (null)", "j")

    fig.suptitle(
        f"Figure 2 — Synthetic Model 1 (Eq. 17)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations  ·  "
        f"red = 1% significance (analytical SE)",
        fontsize=10, fontweight="bold", y=0.995
    )
    return fig


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def make_figure3(store, sig):
    t_ax = np.arange(N - 1)

    fig = plt.figure(figsize=(11, 12))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.09, right=0.97, top=0.92, bottom=0.04)

    # Row 0: causal diagrams
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500 and 1500<t≤2000\n(no causal links)")
    causal_diagram_ax(ax_b, links=[
        (1, 0, "red", False),   # x2->x1
        (1, 2, "red", False),   # x2->x3
        (0, 2, "red", False),   # x1->x3
    ], title="(b) 500<t≤1500\n(x2→x1, x2→x3, x1→x3)")

    # Row 1: (c) biv T_{x2->x1}, (d) tri T_{x2->x1|x3}
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    # Paper's Figure 3 uses a tighter 0-0.08 y-axis for panels (c)-(f) and
    # the wider 0-0.20 axis only for the x1->x3 panels (g)-(h).
    panel_plot(ax_c, t_ax, store["biv_21"].mean(0), sig["biv_21"],
               "|T$_{x2→x1}$|", "c", ylim=0.08)
    panel_plot(ax_d, t_ax, store["tri_21"].mean(0), sig["tri_21"],
               "|T$_{x2→x1|x3}$|", "d", ylim=0.08)

    # Row 2: (e) biv T_{x2->x3}, (f) tri T_{x2->x3|x1}
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    panel_plot(ax_e, t_ax, store["biv_23"].mean(0), sig["biv_23"],
               "|T$_{x2→x3}$|", "e", ylim=0.08)
    panel_plot(ax_f, t_ax, store["tri_23"].mean(0), sig["tri_23"],
               "|T$_{x2→x3|x1}$| (reduced by x1 feedback)", "f", ylim=0.08)

    # Row 3: (g) biv T_{x1->x3}, (h) tri T_{x1->x3|x2}
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])
    panel_plot(ax_g, t_ax, store["biv_13"].mean(0), sig["biv_13"],
               "|T$_{x1→x3}$|", "g", ylim=0.20)
    panel_plot(ax_h, t_ax, store["tri_13"].mean(0), sig["tri_13"],
               "|T$_{x1→x3|x2}$|", "h", ylim=0.20)

    fig.suptitle(
        f"Figure 3 — Synthetic Model 2 (Eq. 18)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations  ·  "
        f"red = 1% significance (analytical SE)",
        fontsize=10, fontweight="bold", y=0.995
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    print(f"Running Synthetic Model 1 ensemble (N_REAL={N_REAL}, N={N})...")
    store1 = run_ensemble(sim_model1)
    sig1 = sig_threshold(store1)

    print(f"\nRunning Synthetic Model 2 ensemble (N_REAL={N_REAL}, N={N})...")
    store2 = run_ensemble(sim_model2)
    sig2 = sig_threshold(store2)

    print("\nGenerating Figure 2...")
    fig2 = make_figure2(store1, sig1)
    path2 = os.path.join(out_dir, "fig2_model1_replication.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print("  Saved: " + path2)
    plt.close(fig2)

    print("Generating Figure 3...")
    fig3 = make_figure3(store2, sig2)
    path3 = os.path.join(out_dir, "fig3_model2_replication.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print("  Saved: " + path3)
    plt.close(fig3)

    # ── Quick verification ─────────────────────────────────────────────────────
    t_ax = np.arange(N - 1)
    coupling_period = (t_ax > 500) & (t_ax <= 1500)
    null_period = t_ax <= 500

    def check(name, arr, expect_sig=True):
        mean_t = arr.mean(0)
        peak = mean_t[coupling_period].mean()
        quiet = mean_t[null_period].mean()
        ratio = peak / (quiet + 1e-12)
        marker = "OK" if (ratio > 2) == expect_sig else "XX"
        print(f"  {marker} {name:35s}  peak/quiet ratio = {ratio:.2f}  peak_val = {mean_t.max():.4f}")

    print("\nVerification (coupling vs null period ratio):")
    print("Model 1:")
    check("biv_12 T_{x1->x2}",       store1["biv_12"], True)
    check("tri_12 T_{x1->x2|x3}",    store1["tri_12"], True)
    check("biv_21 T_{x2->x1}",       store1["biv_21"], True)
    check("biv_23 T_{x2->x3}",       store1["biv_23"], True)
    check("biv_13 T_{x1->x3}",       store1["biv_13"], True)  # spurious but present
    check("tri_13 T_{x1->x3|x2}",    store1["tri_13"], False) # conditioned -> null

    print("Model 2:")
    check("biv_21 T_{x2->x1}",       store2["biv_21"], True)
    check("biv_23 T_{x2->x3}",       store2["biv_23"], True)
    check("biv_13 T_{x1->x3}",       store2["biv_13"], True)
    check("tri_13 T_{x1->x3|x2}",    store2["tri_13"], True)  # direct link remains

    print("\nSig line peak values (model 1):")
    for k in KEYS:
        s = sig1[k]
        print(f"  {k:12s}  sig_peak = {s.max():.4f}")

    print("\nDone.")
