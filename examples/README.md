# Tutorials: replicating Zhou et al. (2024)'s synthetic experiments

Every synthetic-model figure in the paper is replicated here, as both a
standalone script (`python examples/<name>.py`) and a Jupyter notebook
(`examples/notebooks/`). All four use the same `mtvlk` machinery covered in
the [main README](../README.md)'s Quick Start — `compute_mtvlk_filled` with
`qr_mode="adaptive"` and `se_mode="classical"` — so they double as worked
examples of the full pipeline, not just figure-reproduction scripts.

Each script reduces the paper's ensemble size from 1000 realizations to 30
(`N_REAL = 30` near the top of each file) purely for tractability — the
qualitative conclusions (which links are significant, relative peak
magnitudes, the shape of the time-varying curves) are unaffected, only the
curves are a bit noisier than the paper's own smoother 1000-realization
averages. Runtime is roughly 1–5 minutes per script on a normal laptop.

## 1. `fig2_fig3_replication.py` — Synthetic Models 1 & 2

**Notebook:** [`notebooks/01_fig2_fig3_replication.ipynb`](notebooks/01_fig2_fig3_replication.ipynb)

A 3-variable AR(1) system where a set of causal links (e.g. `x1 <-> x2`,
`x2 -> x3`) switches on via a linear ramp, holds, then ramps back off. Two
variants are run:

- **Model 1** (paper Fig. 2, output: [`fig2_model1_replication.png`](fig2_model1_replication.png)):
  bidirectional `x1<->x2` plus `x2->x3`. This one includes the paper's
  demonstration of a *spurious* bivariate link (`x1->x3`, mediated only
  through `x2`) that correctly reads as significant in the naive bivariate
  case but vanishes once you condition on `x2` — the whole point of doing
  multivariate rather than pairwise causality analysis.
- **Model 2** (paper Fig. 3, output: [`fig3_model2_replication.png`](fig3_model2_replication.png)):
  `x2->x1`, `x2->x3`, `x1->x3` with `x2` autonomous.

Both figures compare bivariate vs. conditional (trivariate) estimates of the
same links side by side.

## 2. `fig4_model3_replication.py` — Synthetic Model 3

**Notebook:** [`notebooks/02_fig4_model3_replication.ipynb`](notebooks/02_fig4_model3_replication.ipynb)
**Output:** [`fig4_model3_replication.png`](fig4_model3_replication.png)

A 5-variable network (`x1->x2->x3->x4<-x2`, `x1->x5<-x4`) that switches on
via a ramp and then **stays on** (no ramp-down, unlike Models 1/2) — testing
whether the method correctly tracks a persistent causal structure with
several converging/mediating paths at once. Unlike the first script, every
panel here is already the full 5-variable conditional estimate (conditioning
on all 3 remaining variables each time), since with 5 variables the
bivariate comparison isn't the focus.

## 3. `fig_appendixC_replication.py` — Null-link tests (Appendix C)

**Notebook:** [`notebooks/03_fig_appendixC_replication.ipynb`](notebooks/03_fig_appendixC_replication.ipynb)
**Output:** [`figC1_model1_null.png`](figC1_model1_null.png), [`figC2_model2_null.png`](figC2_model2_null.png), [`figC3_model3_null.png`](figC3_model3_null.png)

For each of the three models above, this plots every pair that has **no**
preset causal link (17 pairs total across the three models) and confirms the
estimated flow stays near/below the significance threshold throughout —
i.e. the method doesn't manufacture false positives. This script imports the
simulators directly from the two scripts above (no re-implementation), so
its results share exactly the same underlying model parameters.

## 4. `synthetic_replication.py` — Supplementary validation

**Notebook:** [`notebooks/04_synthetic_replication.ipynb`](notebooks/04_synthetic_replication.ipynb)
**Output:** [`replication_figure.png`](replication_figure.png)

A separate, self-contained set of validation scenarios not tied to a
specific paper figure: an abrupt (step-function) regime shift, a trivariate
confounder scenario, and a sinusoidally time-varying coupling strength.
Useful as a quick sanity check of the package on cases with a different
character than the paper's own linear ramps.

## Running the notebooks

```bash
pip install "mtvlk[all,notebooks]"
jupyter notebook examples/notebooks/
```

The notebooks are checked in **unexecuted** (no embedded output cells) to
keep the repository small — run them yourself to generate the figures
inline. The pre-generated `.png` files in this directory show what to
expect.
