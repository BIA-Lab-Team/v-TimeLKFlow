"""Replication of Appendix C (Figs. C1-C3) from Zhou et al. (2024) --
"Addition Testing of the Synthetic Model": for each of the three synthetic
models, plot |T| and its significance line for every pair that has NO preset
causality, to confirm the method doesn't produce false positives.

Reuses the exact same synthetic-model simulators and MtvLK configuration as
examples/fig2_fig3_replication.py (models 1/2) and
examples/fig4_model3_replication.py (model 3) -- imported directly rather
than re-defined, to guarantee identical parameters.

Reference: Zhou et al. (2024). J. Climate 37(6), doi:10.1175/JCLI-D-23-0207.1
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mtvlk import compute_mtvlk_filled

import fig2_fig3_replication as f23
import fig4_model3_replication as f4

N_REAL = 30
Z_99 = 2.326
SMOOTH = 50


def _smooth(arr, w):
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:pad + len(arr)]


def panel_plot(ax, t_ax, mean_arr, sig_arr, label, panel_letter, N, ylim=0.20):
    mean_s = _smooth(mean_arr, SMOOTH)
    sig_s = _smooth(sig_arr, SMOOTH)
    ax.fill_between(t_ax, 0, mean_s, alpha=0.20, color="#2166AC")
    ax.plot(t_ax, mean_s, color="#2166AC", lw=1.3, label=label)
    ax.plot(t_ax, sig_s, color="#D73027", lw=1.1, ls="-", label="Sig (1% level)")
    ax.axvline(500, color="black", lw=0.9, ls="--", alpha=0.7)
    if N > 1500:
        ax.axvline(1500, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, N - 1)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("|T| (nats/step)", fontsize=8)
    ax.set_title(f"({panel_letter}) {label}", fontsize=9, pad=3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", handlelength=1.2)


# ── Null-pair ensemble runners (3-variable models 1/2) ───────────────────────

def run_null_ensemble_3var(sim_fn, pairs, init_win, qr_window, fill_window):
    """pairs: dict of name -> (source_idx, target_idx) into the 3x3 T_t/SE_t.
    Runs ONE trivariate compute_mtvlk_filled per realization and extracts all
    requested pairs from it (same call structure as fig2_fig3_replication.py's
    run_ensemble, just extracting different matrix entries)."""
    N = f23.N
    T_len = N - 1
    store = {k: np.zeros((N_REAL, T_len)) for k in pairs}
    store.update({"se_" + k: np.zeros((N_REAL, T_len)) for k in pairs})

    t0 = time.time()
    for r in range(N_REAL):
        rng = np.random.default_rng(r)
        X = sim_fn(rng)
        res = compute_mtvlk_filled(X, dt=f23.DT, Q_scale=f23.Q_SCALE,
                                    R_scale=f23.R_SCALE, init_window=init_win,
                                    qr_mode=f23.QR_MODE, qr_window=qr_window,
                                    fill_window=fill_window, se_mode=f23.SE_MODE,
                                    se_window=f23.SE_WINDOW, return_SE=True)
        T_t, SE_t = res["T_t"], res["SE_t"]
        for name, (j, i) in pairs.items():
            store[name][r] = np.abs(T_t[:, j, i])
            store["se_" + name][r] = SE_t[:, j, i]
        if (r + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    r={r+1:3d}/{N_REAL}  ({elapsed:.0f}s elapsed)")
    return store


def sig_threshold(store, pairs):
    return {k: Z_99 * store["se_" + k].mean(axis=0) / np.sqrt(N_REAL) for k in pairs}


# ── Figure C1 (Synthetic Model 1 null pairs) ─────────────────────────────────

C1_PAIRS = {"c_31_2": (2, 0), "c_32_1": (2, 1)}
C1_LABELS = {
    "c_31_2": r"|T$_{x3\to x1|x2}$|",
    "c_32_1": r"|T$_{x3\to x2|x1}$|",
}


def make_figureC1(store, sig):
    N = f23.N
    t_ax = np.arange(N - 1)
    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.3,
                            left=0.09, right=0.97, top=0.85, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    f23.causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500 and 1500<t≤2000\n(no causal links)")
    f23.causal_diagram_ax(ax_b, links=[(0, 1, "red", True), (1, 2, "red", False)],
                           title="(b) 500<t≤1500\n(x1↔x2, x2→x3)")

    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.3)
    ax_c = fig.add_subplot(row1[0, 0])
    ax_d = fig.add_subplot(row1[0, 1])
    panel_plot(ax_c, t_ax, store["c_31_2"].mean(0), sig["c_31_2"],
               C1_LABELS["c_31_2"], "c", N, ylim=0.20)
    panel_plot(ax_d, t_ax, store["c_32_1"].mean(0), sig["c_32_1"],
               C1_LABELS["c_32_1"], "d", N, ylim=0.20)

    fig.suptitle(
        f"Figure C1 -- Additional testing, Synthetic Model 1 (null links)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations, no preset causality expected",
        fontsize=10, fontweight="bold", y=0.98
    )
    return fig


# ── Figure C2 (Synthetic Model 2 null pairs) ─────────────────────────────────

C2_PAIRS = {"c_12_3": (0, 1), "c_31_2": (2, 0), "c_32_1": (2, 1)}
C2_LABELS = {
    "c_12_3": r"|T$_{x1\to x2|x3}$|",
    "c_31_2": r"|T$_{x3\to x1|x2}$|",
    "c_32_1": r"|T$_{x3\to x2|x1}$|",
}


def make_figureC2(store, sig):
    N = f23.N
    t_ax = np.arange(N - 1)
    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3,
                            left=0.09, right=0.97, top=0.88, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    f23.causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500 and 1500<t≤2000\n(no causal links)")
    f23.causal_diagram_ax(ax_b, links=[(1, 0, "red", False), (1, 2, "red", False), (0, 2, "red", False)],
                           title="(b) 500<t≤1500\n(x2→x1, x2→x3, x1→x3)")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    panel_plot(ax_c, t_ax, store["c_12_3"].mean(0), sig["c_12_3"],
               C2_LABELS["c_12_3"], "c", N, ylim=0.20)
    panel_plot(ax_d, t_ax, store["c_31_2"].mean(0), sig["c_31_2"],
               C2_LABELS["c_31_2"], "d", N, ylim=0.20)
    panel_plot(ax_e, t_ax, store["c_32_1"].mean(0), sig["c_32_1"],
               C2_LABELS["c_32_1"], "e", N, ylim=0.20)

    fig.suptitle(
        f"Figure C2 -- Additional testing, Synthetic Model 2 (null links)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations, no preset causality expected",
        fontsize=10, fontweight="bold", y=0.98
    )
    return fig


# ── Figure C3 (Synthetic Model 3 null pairs, 5-variable) ─────────────────────

C3_PAIRS = {
    "c21_345": (1, 0), "c25_134": (1, 4), "c31_245": (2, 0), "c32_145": (2, 1),
    "c35_124": (2, 4), "c41_235": (3, 0), "c42_135": (3, 1), "c43_125": (3, 2),
    "c51_234": (4, 0), "c52_134": (4, 1), "c53_124": (4, 2), "c54_123": (4, 3),
}
C3_LABELS = {
    "c21_345": r"|T$_{x2\to x1|x3,x4,x5}$|", "c25_134": r"|T$_{x2\to x5|x1,x3,x4}$|",
    "c31_245": r"|T$_{x3\to x1|x2,x4,x5}$|", "c32_145": r"|T$_{x3\to x2|x1,x4,x5}$|",
    "c35_124": r"|T$_{x3\to x5|x1,x2,x4}$|", "c41_235": r"|T$_{x4\to x1|x2,x3,x5}$|",
    "c42_135": r"|T$_{x4\to x2|x1,x3,x5}$|", "c43_125": r"|T$_{x4\to x3|x1,x2,x5}$|",
    "c51_234": r"|T$_{x5\to x1|x2,x3,x4}$|", "c52_134": r"|T$_{x5\to x2|x1,x3,x4}$|",
    "c53_124": r"|T$_{x5\to x3|x1,x2,x4}$|", "c54_123": r"|T$_{x5\to x4|x1,x2,x3}$|",
}
C3_ORDER = ["c21_345", "c25_134", "c31_245", "c32_145", "c35_124", "c41_235",
            "c42_135", "c43_125", "c51_234", "c52_134", "c53_124", "c54_123"]
C3_LETTERS = "cdefghijklmn"


def run_null_ensemble_5var(sim_fn, pairs, init_win, qr_window, fill_window):
    N = f4.N
    T_len = N - 1
    store = {k: np.zeros((N_REAL, T_len)) for k in pairs}
    store.update({"se_" + k: np.zeros((N_REAL, T_len)) for k in pairs})

    t0 = time.time()
    for r in range(N_REAL):
        rng = np.random.default_rng(r)
        X = sim_fn(rng)
        res = compute_mtvlk_filled(X, dt=f4.DT, Q_scale=f4.Q_SCALE,
                                    R_scale=f4.R_SCALE, init_window=init_win,
                                    qr_mode=f4.QR_MODE, qr_window=qr_window,
                                    fill_window=fill_window, se_mode=f4.SE_MODE,
                                    se_window=f4.SE_WINDOW, return_SE=True)
        T_t, SE_t = res["T_t"], res["SE_t"]
        for name, (j, i) in pairs.items():
            store[name][r] = np.abs(T_t[:, j, i])
            store["se_" + name][r] = SE_t[:, j, i]
        if (r + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    r={r+1:3d}/{N_REAL}  ({elapsed:.0f}s elapsed)")
    return store


def make_figureC3(store, sig):
    N = f4.N
    t_ax = np.arange(N - 1)
    fig = plt.figure(figsize=(13, 15))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35,
                            left=0.06, right=0.98, top=0.90, bottom=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    f4.causal_diagram_ax(ax_a, links=[], title="(a) 0<t≤500\n(no causal links)")
    f4.causal_diagram_ax(ax_b, links=f4._ALL_LINKS,
                          title="(b) 500<t≤2000\n(x1→x2→x3→x4←x2, x1→x5←x4)")

    for idx, key in enumerate(C3_ORDER):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        panel_plot(ax, t_ax, store[key].mean(0), sig[key],
                   C3_LABELS[key], C3_LETTERS[idx], N, ylim=0.25)

    fig.suptitle(
        f"Figure C3 -- Additional testing, Synthetic Model 3 (null links)\n"
        f"Ensemble mean of |T| over {N_REAL} realizations, no preset causality expected",
        fontsize=10, fontweight="bold", y=0.97
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    print("Figure C1 (Synthetic Model 1 null pairs)...")
    storeC1 = run_null_ensemble_3var(f23.sim_model1, C1_PAIRS, f23.INIT_WIN,
                                      f23.QR_WINDOW, f23.FILL_WINDOW)
    sigC1 = sig_threshold(storeC1, C1_PAIRS)
    figC1 = make_figureC1(storeC1, sigC1)
    pathC1 = os.path.join(out_dir, "figC1_model1_null.png")
    figC1.savefig(pathC1, dpi=150, bbox_inches="tight")
    print("  Saved:", pathC1)
    plt.close(figC1)

    print("\nFigure C2 (Synthetic Model 2 null pairs)...")
    storeC2 = run_null_ensemble_3var(f23.sim_model2, C2_PAIRS, f23.INIT_WIN,
                                      f23.QR_WINDOW, f23.FILL_WINDOW)
    sigC2 = sig_threshold(storeC2, C2_PAIRS)
    figC2 = make_figureC2(storeC2, sigC2)
    pathC2 = os.path.join(out_dir, "figC2_model2_null.png")
    figC2.savefig(pathC2, dpi=150, bbox_inches="tight")
    print("  Saved:", pathC2)
    plt.close(figC2)

    print("\nFigure C3 (Synthetic Model 3 null pairs)...")
    storeC3 = run_null_ensemble_5var(f4.sim_model3, C3_PAIRS, f4.INIT_WIN,
                                      f4.QR_WINDOW, f4.FILL_WINDOW)
    sigC3 = sig_threshold(storeC3, C3_PAIRS)
    figC3 = make_figureC3(storeC3, sigC3)
    pathC3 = os.path.join(out_dir, "figC3_model3_null.png")
    figC3.savefig(pathC3, dpi=150, bbox_inches="tight")
    print("  Saved:", pathC3)
    plt.close(figC3)

    # ── Quick verification: all null pairs should stay non-significant ──────
    print("\nVerification (all should be non-significant, i.e. blue <~ red):")

    def check_null(name, arr, sig_arr):
        mean_t = arr.mean(0)
        peak = mean_t.max()
        sig_peak_at_peak_t = sig_arr[mean_t.argmax()]
        marker = "OK" if peak <= 1.5 * sig_peak_at_peak_t else "XX"
        print(f"  {marker} {name:12s}  peak|T|={peak:.4f}  sig_at_peak={sig_peak_at_peak_t:.4f}")

    for name in C1_PAIRS:
        check_null("C1:" + name, storeC1[name], sigC1[name])
    for name in C2_PAIRS:
        check_null("C2:" + name, storeC2[name], sigC2[name])
    for name in C3_PAIRS:
        check_null("C3:" + name, storeC3[name], sigC3[name])

    print("\nDone.")
