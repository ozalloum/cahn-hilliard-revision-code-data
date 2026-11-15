# Supplementary Material S2: step-conditioned correction experiment

**Author:** Othman H. Y. Zalloum

**Affiliation:** Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

Code, frozen coefficients and validation data for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**.

## Contents

- experiment_step_conditioned.py: training, ETDRK4 reference integration,
  macro-step transfer comparisons and structural checks.
- repro_code/numerics/validation/kernel.py: the author's public numerical
  validation kernel, copied from
  https://github.com/ozalloum/cahn-hilliard-revision-code-data.
- make_step_transfer_figure.py: plotting source for Figure 9.
- results/exploratory_filters.npz: frozen field (R) and increment (B)
  multipliers fitted from 600 training trajectories times 16 states.
- results/per_case.json: all 100 held-out trajectories, four time steps
  and five methods, including counts of steps invoking energy backtracking.
- results/summary.json, run_metadata.json, structural_checks.json:
  numerical results and checks.
- Figure_9.pdf and Figure_9.png: direct plots of the supplied data.

## Numerical protocol

All tests use N=64, epsilon=0.02, M=1, S=2,
training step h0=1e-4, final time T=0.0035,
float64/complex128 arithmetic and original random seeds.
Pairs use the author's 100-substep teacher and six-substep propagator.
Both regressions use summed complex ridge penalty lambda=1e-12
and the historical floating-point radius-28 mask, excluding four
axial nominal-radius-28 modes.

The independent test reference is ETDRK4 with step 1.25e-5.
It is halved to 6.25e-6 for two test fields; the observed maximum
relative difference is 1.3464150846282161e-13.

## Metric names and archived diagnostics

New evaluations use `final_mass_error` for
`abs(mean(u_final) - mean(u_initial))`. In the supplied historical
`results/per_case.json`, the key `max_mass_error` contains this same
final-only quantity; it is not a maximum over the trajectory. The historical
JSON is retained unchanged. Readers supporting both versions should interpret
the old key as an alias for `final_mass_error`.

The archived `backtracks` value counts macro steps with safeguard code > 0,
including a code-35 fallback. It is not the sum of halving indices. The archived
S2 records do not retain accepted fractions, fallback counts or the kernel's
post-correction acceptance-violation flag separately.

## Rerun

Install Python 3 with numpy (results used NumPy 2.3.5);
install matplotlib to regenerate the figure.

To reevaluate the archived coefficients:

~~~bash
python experiment_step_conditioned.py \
  --reuse-filters results/exploratory_filters.npz \
  --train 600 --snapshots 16 --test 100 \
  --dts 0.0001 0.00005 0.000025 0.0000125 \
  --output reevaluated_results
~~~

To refit from all 9,600 pairs, omit --reuse-filters:

~~~bash
python experiment_step_conditioned.py \
  --train 600 --snapshots 16 --test 100 \
  --dts 0.0001 0.00005 0.000025 0.0000125 \
  --output independently_regenerated
~~~

Regenerate the plot from supplied summary:

~~~bash
python make_step_transfer_figure.py results/summary.json Figure_9.pdf
~~~

## Interpretation and limitations

The script implements a new increment-anchored modal fit B(k)
and a step factor h/h0, plus a simple (h/h0)^2-tapered original
field filter as an ablation. This ensures the unsafeguarded increment
correction is O(h^2) for bounded states and inherits fixed points
of the base propagator.
It does not prove long-time stability, unconditional energy dissipation
or physical-time progress after backtracking. Twelve uncorrected
substeps remain more accurate on the measured 100-field ensemble.
The experiment is restricted to the original unpadded 64-by-64
collocation system, not a spatial-convergence test.
Timings depend on machine and runtime.
