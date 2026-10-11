Historical submitted snapshot. See the top-level README.txt and
numerics/validation/ for the revised evidence and conclusions.

# Reproducibility package

## Manuscript
A Structure-Preserving Spectral-Residual Surrogate for Accelerated Cahn--Hilliard Simulation

## Main executable
Run:

    python reproduce_cahn_hilliard.py

The script regenerates the spectral-residual fit, the 100-trajectory test benchmark, long-horizon tests, epsilon and mobility sweeps, the unprojected baseline, timing results, figures, and CSV outputs.

## Dependencies
Install the packages listed in `requirements.txt`.

## Fixed numerical configuration
- Grid: 64 x 64 periodic grid on [0, 2pi]^2
- Seed: 20260618
- Interface width epsilon: 0.020
- Mobility: 1.0
- Stabilization: 2.0
- Macro step: 1e-4
- Reference substeps per macro step: 100
- Surrogate coarse substeps per macro step: 6
- Fitting trajectories: 600
- One-step fitting states per trajectory: 16
- Independent test trajectories: 100
- Main rollout: 35 macro steps
- Long-horizon rollouts: 350 and 700 macro steps on 10 test trajectories
- Retained spectral cutoff: |k| <= 28
- Structure-factor radial bins: 32

## Data split clarification
The original Colab notebook used legacy labels corresponding to 500 training and 100 validation trajectories. In the executed workflow, both groups enter the final least-squares spectral-filter fit, so the manuscript and public script correctly describe them as 600 fitting trajectories. The 100 test trajectories remain independent and are not used for fitting.

## Timing note
The script performs one warm-up step for each method and averages three complete 35-step rollout timings. Absolute wall-clock values depend on hardware and software environment; the manuscript therefore reports the archived timing values while emphasizing the relative speed-up.

## Scope of reported OOD results
Only numerically generated epsilon and mobility sweeps are included in this public package and reported in the manuscript.
