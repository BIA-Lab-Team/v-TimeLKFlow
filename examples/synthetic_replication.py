"""Synthetic replication of Zhou et al. (2024) MtvLK validation experiments.

Three experiments:
  1. Bivariate abrupt regime shift     — MtvLK tracks step change; static LK averages it
  2. Trivariate confounded system      — multivariate LK removes spurious bivariate signal
  3. Bivariate sinusoidal coupling     — MtvLK tracks continuous oscillation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mtvlk import compute_lk, compute_mtvlk

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic model generators
# ─────────────────────────────────────────────────────────────────────────────

def exp1_abrupt(N=1000, sigma=0.3, seed=0):
    """Bivariate AR(1) with coupling step at N//2."""
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 2))
    mid = N // 2
    for t in range(N - 1):
        c = 0.0 if t < mid else 0.5
        X[t+1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
        X[t+1, 1] = c   * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()
    c_true = np.where(np.arange(N-1) < mid, 0.0, 0.5)
    return X, c_true


def exp2_confounded(N=3000, sigma=0.3, seed=1):
    """Trivariate: x0 drives x1 and x2 independently (shared driver = confounder)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 3))
    for t in range(N - 1):
        X[t+1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
        X[t+1, 1] = 0.4 * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()
        X[t+1, 2] = 0.4 * X[t, 0] + 0.3 * X[t, 2] + sigma * rng.standard_normal()
    return X


def exp3_sinusoidal(N=1500, sigma=0.3, period=500, seed=2):
    """Bivariate AR(1) with sinusoidally varying positive coupling.

    c(t) = 0.25 + 0.25*sin(2π t/period), ranging from 0 to 0.5.
    Always positive avoids sign cancellation; long period gives the KF time to track.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 2))
    t_arr = np.arange(N)
    c_arr = 0.25 + 0.25 * np.sin(2 * np.pi * t_arr / period)
    for t in range(N - 1):
        X[t+1, 0] = 0.5 * X[t, 0] + sigma * rng.standard_normal()
        X[t+1, 1] = c_arr[t] * X[t, 0] + 0.3 * X[t, 1] + sigma * rng.standard_normal()
    c_true = c_arr[:-1]  # align with T_t (length N-1)
    return X, c_true


# ─────────────────────────────────────────────────────────────────────────────
# Run experiments
# ─────────────────────────────────────────────────────────────────────────────

print("Running Experiment 1: Abrupt regime shift …")
X1, c_true1 = exp1_abrupt()
res1_static = compute_lk(X1, dt=1.0, n_boot=0)
res1_tv     = compute_mtvlk(X1, dt=1.0, Q_scale=5e-5, R_scale=1.0, init_window=50)

print("Running Experiment 2: Confounded trivariate system …")
X2 = exp2_confounded()
# Bivariate LK: pairwise grid (use tau)
n2 = 3
tau_biv = np.full((n2, n2), np.nan)
for src in range(n2):
    for tgt in range(n2):
        if src == tgt:
            continue
        res_pair = compute_lk(X2[:, [src, tgt]], dt=1.0, n_boot=0)
        # res_pair["tau"] is (2,2); T[0,1] = tau_{src->tgt} in pair indexing
        tau_biv[src, tgt] = res_pair["tau"][0, 1]
res2_multi = compute_lk(X2, dt=1.0, n_boot=0)

print("Running Experiment 3: Sinusoidal coupling …")
X3, c_true3 = exp3_sinusoidal()
res3_tv = compute_mtvlk(X3, dt=1.0, Q_scale=5e-4, R_scale=1.0, init_window=50)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

BLUE   = "#2C7BB6"
RED    = "#D7191C"
GREEN  = "#1A9641"
ORANGE = "#FDAE61"
GRAY   = "#666666"

fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    3, 2, figure=fig,
    hspace=0.55, wspace=0.38,
    left=0.08, right=0.96, top=0.94, bottom=0.06
)

t1   = np.arange(len(X1))
t1_d = np.arange(len(res1_tv["T_t"]))   # N-1
mid  = len(X1) // 2

# ── Panel A: time series (Exp 1) ────────────────────────────────────────────
ax_ts = fig.add_subplot(gs[0, 0])
ax_ts.plot(t1, X1[:, 0], color=BLUE,   lw=0.8, alpha=0.85, label="$x_0$")
ax_ts.plot(t1, X1[:, 1], color=ORANGE, lw=0.8, alpha=0.85, label="$x_1$")
ax_ts.axvline(mid, color=GRAY, ls="--", lw=1.2, label="Regime shift")
ax_ts.set_xlabel("Time step")
ax_ts.set_ylabel("Value")
ax_ts.set_title("(a) Exp 1 — Time series", fontweight="bold", fontsize=10)
ax_ts.legend(fontsize=8, loc="upper left")

# ── Panel B: MtvLK vs static (Exp 1) ────────────────────────────────────────
ax_lk = fig.add_subplot(gs[0, 1])
T_tv1 = res1_tv["T_t"][:, 0, 1]          # T_{x0->x1}(t)
T_st1 = res1_static["T"][0, 1]            # static scalar

# rolling mean to reduce noise for display
win = 30
T_tv1_smooth = np.convolve(T_tv1, np.ones(win)/win, mode="same")

ax_lk.plot(t1_d, c_true1, color=GREEN,  lw=1.5, ls="-",  label="True $c(t)$", zorder=3)
ax_lk.plot(t1_d, T_tv1_smooth, color=BLUE,  lw=1.5, label="MtvLK $T_{x_0\\!\\to\\!x_1}(t)$")
ax_lk.axhline(T_st1, color=RED, lw=1.5, ls="--", label=f"Static LK = {T_st1:.3f}")
ax_lk.axvline(mid, color=GRAY, ls="--", lw=1.0)
ax_lk.axhline(0, color="black", lw=0.6, ls=":")
ax_lk.set_xlabel("Time step")
ax_lk.set_ylabel("$T$ or $c(t)$")
ax_lk.set_title("(b) Exp 1 — MtvLK vs static LK", fontweight="bold", fontsize=10)
ax_lk.legend(fontsize=8, loc="upper left")

# ── Panel C: bivariate LK heatmap (Exp 2) ───────────────────────────────────
ax_biv = fig.add_subplot(gs[1, 0])
labels2 = ["$x_0$", "$x_1$", "$x_2$"]
vmax2 = np.nanmax(np.abs(tau_biv))
im_biv = ax_biv.imshow(tau_biv.T, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
ax_biv.set_xticks([0, 1, 2]); ax_biv.set_xticklabels(labels2)
ax_biv.set_yticks([0, 1, 2]); ax_biv.set_yticklabels(labels2)
ax_biv.set_xlabel("Source"); ax_biv.set_ylabel("Target")
ax_biv.set_title("(c) Exp 2 — Bivariate LK $\\tau$ [%]", fontweight="bold", fontsize=10)
for src in range(3):
    for tgt in range(3):
        v = tau_biv[src, tgt]
        if not np.isnan(v):
            ax_biv.text(src, tgt, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(v) > 0.6*vmax2 else "black")
plt.colorbar(im_biv, ax=ax_biv, label="%", fraction=0.046, pad=0.04)

# ── Panel D: multivariate LK heatmap (Exp 2) ────────────────────────────────
ax_mv = fig.add_subplot(gs[1, 1])
tau_mv = res2_multi["tau"]
vmax_mv = np.nanmax(np.abs(tau_mv))
im_mv = ax_mv.imshow(tau_mv.T, cmap="RdBu_r", vmin=-vmax_mv, vmax=vmax_mv, aspect="auto")
ax_mv.set_xticks([0, 1, 2]); ax_mv.set_xticklabels(labels2)
ax_mv.set_yticks([0, 1, 2]); ax_mv.set_yticklabels(labels2)
ax_mv.set_xlabel("Source"); ax_mv.set_ylabel("Target")
ax_mv.set_title("(d) Exp 2 — Multivariate LK $\\tau$ [%]", fontweight="bold", fontsize=10)
for src in range(3):
    for tgt in range(3):
        v = tau_mv[src, tgt]
        ax_mv.text(src, tgt, f"{v:.1f}", ha="center", va="center",
                   fontsize=9, color="white" if abs(v) > 0.6*vmax_mv else "black")
plt.colorbar(im_mv, ax=ax_mv, label="%", fraction=0.046, pad=0.04)

# ── Panel E: sinusoidal coupling (Exp 3) ─────────────────────────────────────
ax_sin = fig.add_subplot(gs[2, :])
t3_d = np.arange(len(res3_tv["T_t"]))
T_tv3_01 = res3_tv["T_t"][:, 0, 1]   # T_{x0->x1}(t)
T_tv3_10 = res3_tv["T_t"][:, 1, 0]   # T_{x1->x0}(t)  — should be ~0
win3 = 20
T_tv3_01_s = np.convolve(T_tv3_01, np.ones(win3)/win3, mode="same")
T_tv3_10_s = np.convolve(T_tv3_10, np.ones(win3)/win3, mode="same")

ax_sin.plot(t3_d, c_true3, color=GREEN,  lw=1.8, label="True $c(t)$", zorder=4)
ax_sin.plot(t3_d, T_tv3_01_s, color=BLUE,  lw=1.5, label="MtvLK $T_{x_0\\!\\to\\!x_1}(t)$")
ax_sin.plot(t3_d, T_tv3_10_s, color=RED,   lw=1.2, ls="--", alpha=0.7,
            label="MtvLK $T_{x_1\\!\\to\\!x_0}(t)$ (null direction)")
ax_sin.axhline(0, color="black", lw=0.6, ls=":")
ax_sin.set_xlabel("Time step")
ax_sin.set_ylabel("$T$ or $c(t)$")
ax_sin.set_title("(e) Exp 3 — Sinusoidal coupling: MtvLK tracks continuous variation",
                 fontweight="bold", fontsize=10)
ax_sin.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Synthetic replication of Zhou et al. (2024) — Multivariate Time-Varying LK Information Flow",
    fontsize=11, fontweight="bold", y=0.98
)

out_path = os.path.join(os.path.dirname(__file__), "replication_figure.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────────────
half = (len(X1) - 1) // 2
T_before = np.nanmean(res1_tv["T_t"][:half, 0, 1])
T_after  = np.nanmean(res1_tv["T_t"][half:, 0, 1])
assert T_after > T_before, f"Exp1: expected T_after ({T_after:.4f}) > T_before ({T_before:.4f})"
print(f"Exp 1 ✓  T_before={T_before:.4f}, T_after={T_after:.4f}")

tau_biv_12  = abs(tau_biv[1, 2])   # bivariate x1->x2
tau_mv_12   = abs(tau_mv[1, 2])    # multivariate x1->x2
assert tau_mv_12 < tau_biv_12, (
    f"Exp2: expected multivariate |τ_12|={tau_mv_12:.2f} < bivariate={tau_biv_12:.2f}"
)
print(f"Exp 2 ✓  bivariate |τ(x1→x2)|={tau_biv_12:.2f}%, multivariate={tau_mv_12:.2f}%")

valid = np.isfinite(T_tv3_01)
corr = np.corrcoef(c_true3[valid], T_tv3_01[valid])[0, 1]
assert corr > 0.3, f"Exp3: correlation between c(t) and MtvLK={corr:.3f} < 0.3"
print(f"Exp 3 ✓  correlation(c(t), MtvLK)={corr:.3f}")

print("\nAll verification checks passed.")
