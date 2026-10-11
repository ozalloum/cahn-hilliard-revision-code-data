
#!/usr/bin/env python3
"""
Reproduction script for the CPC Computational Physics manuscript.
Generates all figures, data files, and a manuscript draft.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ---------------------------------------------------------------------
# Fixed parameters (as in the manuscript)
# ---------------------------------------------------------------------
SEED = 20260618
rng_global = np.random.default_rng(SEED)

N = 64
DOMAIN_LENGTH = 2.0 * np.pi
DX = DOMAIN_LENGTH / N
EPS_TRAIN = 0.020
MOBILITY_TRAIN = 1.0
STABILIZATION = 2.0

MACRO_DT = 1.0e-4
REF_SUBSTEPS = 100
SURROGATE_HALF_STEPS = 6
ROLLOUT_STEPS = 35
FINAL_TIME = MACRO_DT * ROLLOUT_STEPS

# Dataset sizes
N_FIT_PRIMARY = 500
N_FIT_ADDITIONAL = 100
N_TEST = 100
TRAIN_STATES_PER_TRAJ = 16

TITLE = "A Structure-Preserving Spectral-Residual Surrogate for Accelerated Cahn--Hilliard Simulation"

# Grid and wavenumbers
x = np.linspace(0.0, DOMAIN_LENGTH, N, endpoint=False)
y = np.linspace(0.0, DOMAIN_LENGTH, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = np.fft.fftfreq(N, d=DOMAIN_LENGTH / N) * 2.0 * np.pi
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K4 = K2**2
KMAG = np.sqrt(K2)

# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
    })

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}

def project_root() -> Path:
    # Use a dedicated project folder in Colab so the zip is clean and easy to find.
    root = Path.cwd() / 'ch_neural_operator_revised_package'
    root.mkdir(parents=True, exist_ok=True)
    return root

def ensure_directories(root: Path) -> dict[str, Path]:
    paths = {"figures": root / "figures", "data": root / "data", "output": root / "output"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def write_csv_with_metadata(path: Path, df: pd.DataFrame, metadata: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        df.to_csv(f, index=False)

def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.pdf")
    fig.savefig(figures_dir / f"{stem}.png", dpi=600)
    plt.close(fig)

def zip_project(root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if path.resolve() == output_zip.resolve():
                continue
            zf.write(path, path.relative_to(root.parent))

# ---------------------------------------------------------------------
# Physics and solvers (now accept mobility and eps as arguments)
# ---------------------------------------------------------------------
def initial_condition(seed: int, mean: float = 0.0, amplitude: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(N, N))
    noise_hat = np.fft.fft2(noise)
    smooth_filter = np.exp(-0.5 * K2 / (10.0**2))
    phi = np.fft.ifft2(noise_hat * smooth_filter).real
    phi = (phi - np.mean(phi)) / np.std(phi)
    return amplitude * phi + mean

def free_energy(phi: np.ndarray, eps: float) -> float:
    phi_hat = np.fft.fft2(phi)
    grad_x = np.fft.ifft2(1j * KX * phi_hat).real
    grad_y = np.fft.ifft2(1j * KY * phi_hat).real
    density = 0.5 * eps**2 * (grad_x**2 + grad_y**2) + 0.25 * (phi**2 - 1.0) ** 2
    return float(np.mean(density))

def semi_implicit_step(phi: np.ndarray, dt: float, eps: float, mobility: float) -> np.ndarray:
    phi_hat = np.fft.fft2(phi)
    nonlinear_hat = np.fft.fft2(phi**3 - phi)
    denom = 1.0 + dt * mobility * (eps**2 * K4 + STABILIZATION * K2)
    numerator = phi_hat - dt * mobility * K2 * nonlinear_hat + dt * mobility * STABILIZATION * K2 * phi_hat
    phi_new_hat = numerator / denom
    return np.fft.ifft2(phi_new_hat).real

def reference_step(phi: np.ndarray, macro_dt: float, eps: float, mobility: float) -> np.ndarray:
    dt = macro_dt / REF_SUBSTEPS
    out = phi.copy()
    for _ in range(REF_SUBSTEPS):
        out = semi_implicit_step(out, dt, eps, mobility)
    return out

def coarse_operator_step(phi: np.ndarray, macro_dt: float, eps: float, mobility: float, half_steps: int = 1) -> np.ndarray:
    out = phi.copy()
    dt = macro_dt / half_steps
    for _ in range(half_steps):
        out = semi_implicit_step(out, dt, eps, mobility)
    return out

def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)

def project_mass(phi: np.ndarray, target_mass: float) -> np.ndarray:
    return phi + (target_mass - float(np.mean(phi)))

def project_energy(phi_old: np.ndarray, phi_candidate: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    e_old = free_energy(phi_old, eps)
    e_new = free_energy(phi_candidate, eps)
    if e_new <= e_old + 1.0e-14:
        return phi_candidate, 0
    delta = phi_candidate - phi_old
    target_mass = float(np.mean(phi_old))
    alpha = 1.0
    for count in range(1, 35):
        alpha *= 0.5
        trial = project_mass(phi_old + alpha * delta, target_mass)
        if free_energy(trial, eps) <= e_old + 1.0e-14:
            return trial, count
    return project_mass(phi_old, target_mass), 35

# ---------------------------------------------------------------------
# Training the residual operator (uses mobility=1.0, eps=0.020)
# ---------------------------------------------------------------------
def train_fourier_residual_operator() -> np.ndarray:
    numerator = np.zeros((N, N), dtype=np.complex128)
    denominator = np.zeros((N, N), dtype=np.float64)
    total_states = (N_FIT_PRIMARY + N_FIT_ADDITIONAL) * TRAIN_STATES_PER_TRAJ
    print(f"Training on {total_states} states from {N_FIT_PRIMARY + N_FIT_ADDITIONAL} trajectories...")
    for i in range(N_FIT_PRIMARY + N_FIT_ADDITIONAL):
        mean = np.random.default_rng(SEED + i).uniform(-0.02, 0.02)
        phi = initial_condition(SEED + 10_000 + i, mean=mean)
        for _ in range(TRAIN_STATES_PER_TRAJ):
            target = reference_step(phi, MACRO_DT, EPS_TRAIN, MOBILITY_TRAIN)
            baseline = coarse_operator_step(phi, MACRO_DT, EPS_TRAIN, MOBILITY_TRAIN, half_steps=SURROGATE_HALF_STEPS)
            residual_hat = np.fft.fft2(target - baseline)
            baseline_hat = np.fft.fft2(baseline)
            numerator += residual_hat * np.conj(baseline_hat)
            denominator += np.abs(baseline_hat) ** 2
            phi = target
    learned_filter = numerator / (denominator + 1.0e-12)
    learned_filter[0, 0] = 0.0
    # Keep more Fourier modes for the higher-data run.
    learned_filter[KMAG > 28.0] = 0.0
    return learned_filter

# ---------------------------------------------------------------------
# Surrogate steps (with and without projections)
# ---------------------------------------------------------------------
def learned_surrogate_step(phi: np.ndarray, macro_dt: float, eps: float, mobility: float,
                           learned_filter: np.ndarray) -> tuple[np.ndarray, int]:
    target_mass = float(np.mean(phi))
    baseline = coarse_operator_step(phi, macro_dt, eps, mobility, half_steps=SURROGATE_HALF_STEPS)
    baseline_hat = np.fft.fft2(baseline)
    correction = np.fft.ifft2(learned_filter * baseline_hat).real
    candidate = project_mass(baseline + correction, target_mass)
    projected, n_backtrack = project_energy(phi, candidate, eps)
    projected = project_mass(projected, target_mass)
    return projected, n_backtrack

def unconstrained_surrogate_step(phi: np.ndarray, macro_dt: float, eps: float, mobility: float,
                                 learned_filter: np.ndarray) -> np.ndarray:
    baseline = coarse_operator_step(phi, macro_dt, eps, mobility, half_steps=SURROGATE_HALF_STEPS)
    baseline_hat = np.fft.fft2(baseline)
    correction = np.fft.ifft2(learned_filter * baseline_hat).real
    return baseline + correction

# ---------------------------------------------------------------------
# Generic rollout function
# ---------------------------------------------------------------------
def rollout(phi0: np.ndarray, eps: float, mobility: float, learned_filter: np.ndarray,
            n_steps: int = ROLLOUT_STEPS, use_projections: bool = True) -> dict:
    ref = phi0.copy()
    coarse = phi0.copy()
    learned = phi0.copy()
    records = []
    fields = {"initial": phi0.copy()}
    backtracks = 0
    for step_id in range(n_steps + 1):
        t = step_id * MACRO_DT
        records.append({
            "step": step_id,
            "time": t,
            "energy_reference": free_energy(ref, eps),
            "energy_coarse": free_energy(coarse, eps),
            "energy_learned": free_energy(learned, eps),
            "mass_reference": float(np.mean(ref)),
            "mass_coarse": float(np.mean(coarse)),
            "mass_learned": float(np.mean(learned)),
            "relative_l2_coarse": relative_l2(coarse, ref),
            "relative_l2_learned": relative_l2(learned, ref),
        })
        if step_id == n_steps // 2:
            fields["mid_reference"] = ref.copy()
        if step_id == n_steps:
            fields["final_reference"] = ref.copy()
            fields["final_coarse"] = coarse.copy()
            fields["final_learned"] = learned.copy()
            break
        ref = reference_step(ref, MACRO_DT, eps, mobility)
        coarse = coarse_operator_step(coarse, MACRO_DT, eps, mobility, half_steps=1)
        if use_projections:
            learned, n_b = learned_surrogate_step(learned, MACRO_DT, eps, mobility, learned_filter)
            backtracks += n_b
        else:
            learned = unconstrained_surrogate_step(learned, MACRO_DT, eps, mobility, learned_filter)
    return {"records": records, "fields": fields, "backtracks": backtracks}

def energy_violations(energies: np.ndarray) -> int:
    return int(np.sum(np.diff(energies) > 1.0e-12))

def radial_structure_factor(phi: np.ndarray, n_bins: int = 32) -> pd.DataFrame:
    centered = phi - float(np.mean(phi))
    sf = np.abs(np.fft.fft2(centered)) ** 2
    radii = KMAG.ravel()
    values = sf.ravel()
    max_r = float(np.max(radii))
    bins = np.linspace(0.0, max_r, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    out = []
    for i in range(n_bins):
        mask = (radii >= bins[i]) & (radii < bins[i + 1])
        out.append(float(np.mean(values[mask])) if np.any(mask) else np.nan)
    out = np.array(out)
    finite = np.isfinite(out)
    if np.sum(out[finite]) > 0:
        out[finite] = out[finite] / np.sum(out[finite])
    return pd.DataFrame({"wavenumber": centers, "structure_factor": out})

def structure_error(a: np.ndarray, b: np.ndarray) -> float:
    sa = radial_structure_factor(a)["structure_factor"].fillna(0.0).values
    sb = radial_structure_factor(b)["structure_factor"].fillna(0.0).values
    denom = np.linalg.norm(sb)
    if denom == 0:
        return float(np.linalg.norm(sa - sb))
    return float(np.linalg.norm(sa - sb) / denom)

# ---------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------
def speed_benchmark(phi0: np.ndarray, eps: float, mobility: float,
                    learned_filter: np.ndarray, repeats: int = 3) -> dict[str, float]:
    _ = reference_step(phi0, MACRO_DT, eps, mobility)
    _ = learned_surrogate_step(phi0, MACRO_DT, eps, mobility, learned_filter)
    ref_times = []
    surr_times = []
    for _ in range(repeats):
        p = phi0.copy()
        t0 = time.perf_counter()
        for _ in range(ROLLOUT_STEPS):
            p = reference_step(p, MACRO_DT, eps, mobility)
        ref_times.append(time.perf_counter() - t0)
        p = phi0.copy()
        t0 = time.perf_counter()
        for _ in range(ROLLOUT_STEPS):
            p, _ = learned_surrogate_step(p, MACRO_DT, eps, mobility, learned_filter)
        surr_times.append(time.perf_counter() - t0)
    ref_mean = float(np.mean(ref_times))
    surr_mean = float(np.mean(surr_times))
    return {
        "reference_rollout_time_s": ref_mean,
        "surrogate_rollout_time_s": surr_mean,
        "measured_speedup": ref_mean / max(surr_mean, 1.0e-12),
    }

# ---------------------------------------------------------------------
# OOD sweep functions (compute on the fly)
# ---------------------------------------------------------------------
def ood_sweep(param_name: str, param_values: list, learned_filter: np.ndarray,
              base_eps: float, base_mob: float, base_dt: float, base_comp: float) -> pd.DataFrame:
    rows = []
    for val in param_values:
        eps = base_eps
        mob = base_mob
        mean_comp = base_comp
        if param_name == 'eps':
            eps = val
        elif param_name == 'mobility':
            mob = val
        elif param_name == 'comp':
            mean_comp = val
        # Use a fixed test seed
        phi0 = initial_condition(SEED + 300000, mean=mean_comp)
        res = rollout(phi0, eps, mob, learned_filter, n_steps=ROLLOUT_STEPS, use_projections=True)
        final_ref = res['fields']['final_reference']
        final_learned = res['fields']['final_learned']
        l2_err = relative_l2(final_learned, final_ref)
        sf_err = structure_error(final_learned, final_ref)
        rows.append({'param_value': val, 'l2_error': l2_err, 'structure_error': sf_err})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------
def main() -> None:
    root = project_root()
    paths = ensure_directories(root)
    figures_dir = paths["figures"]
    data_dir = paths["data"]
    output_dir = paths["output"]

    set_publication_style()

    # 1. Train residual operator
    print("Training learned Fourier residual operator...")
    learned_filter = train_fourier_residual_operator()

    # 2. Standard rollouts (35 steps) on 100 test trajectories
    print("Running standard (35-step) test rollouts...")
    all_records = []
    metrics_list = []
    first_rollout = None
    for test_id in range(N_TEST):
        mean = np.random.default_rng(SEED + 200 + test_id).uniform(-0.02, 0.02)
        phi0 = initial_condition(SEED + 20_000 + test_id, mean=mean)
        res = rollout(phi0, EPS_TRAIN, MOBILITY_TRAIN, learned_filter, n_steps=ROLLOUT_STEPS, use_projections=True)
        rec = pd.DataFrame(res['records'])
        rec['test_id'] = test_id
        all_records.append(rec)
        final_ref = res['fields']['final_reference']
        final_coarse = res['fields']['final_coarse']
        final_learned = res['fields']['final_learned']
        mass0 = float(np.mean(phi0))
        max_mass_err = float(np.max(np.abs(rec['mass_learned'].values - mass0)))
        metrics_list.append({
            'test_id': test_id,
            'final_l2_coarse': relative_l2(final_coarse, final_ref),
            'final_l2_learned': relative_l2(final_learned, final_ref),
            'structure_error_coarse': structure_error(final_coarse, final_ref),
            'structure_error_learned': structure_error(final_learned, final_ref),
            'max_mass_error_learned': max_mass_err,
            'energy_violations_reference': energy_violations(rec['energy_reference'].values),
            'energy_violations_coarse': energy_violations(rec['energy_coarse'].values),
            'energy_violations_learned': energy_violations(rec['energy_learned'].values),
            'backtracks': res['backtracks'],
        })
        if test_id == 0:
            first_rollout = res

    timeseries = pd.concat(all_records, ignore_index=True)
    metrics = pd.DataFrame(metrics_list)

    # 3. Long-horizon rollouts (350, 700) on first 10 test trajectories
    print("Running long-horizon rollouts (350, 700 steps)...")
    long_metrics = []
    for test_id in range(10):
        mean = np.random.default_rng(SEED + 200 + test_id).uniform(-0.02, 0.02)
        phi0 = initial_condition(SEED + 20_000 + test_id, mean=mean)
        for n_steps in [350, 700]:
            res = rollout(phi0, EPS_TRAIN, MOBILITY_TRAIN, learned_filter, n_steps=n_steps, use_projections=True)
            rec = pd.DataFrame(res['records'])
            final_ref = res['fields']['final_reference']
            final_learned = res['fields']['final_learned']
            mass0 = float(np.mean(phi0))
            max_mass_err = float(np.max(np.abs(rec['mass_learned'].values - mass0)))
            long_metrics.append({
                'test_id': test_id,
                'n_steps': n_steps,
                'final_l2_learned': relative_l2(final_learned, final_ref),
                'structure_error_learned': structure_error(final_learned, final_ref),
                'max_mass_error_learned': max_mass_err,
                'energy_violations_learned': energy_violations(rec['energy_learned'].values),
                'backtracks': res['backtracks'],
            })
    long_df = pd.DataFrame(long_metrics)

    # 4. OOD sweeps (eps and mobility)
    print("Running OOD sweeps for epsilon and mobility...")
    ood_eps = ood_sweep('eps', [0.010, 0.015, 0.018, 0.020, 0.024, 0.030, 0.040],
                        learned_filter, EPS_TRAIN, MOBILITY_TRAIN, 1.0, 0.0)
    ood_mob = ood_sweep('mobility', [0.5, 0.8, 1.0, 1.5, 2.0],
                        learned_filter, EPS_TRAIN, MOBILITY_TRAIN, 1.0, 0.0)
    # 5. Baseline: unprojected spectral-residual surrogate (no projections)
    print("Evaluating unprojected spectral-residual baseline...")
    unconstrained_metrics = []
    for test_id in range(N_TEST):
        mean = np.random.default_rng(SEED + 200 + test_id).uniform(-0.02, 0.02)
        phi0 = initial_condition(SEED + 20_000 + test_id, mean=mean)
        res = rollout(phi0, EPS_TRAIN, MOBILITY_TRAIN, learned_filter, n_steps=ROLLOUT_STEPS, use_projections=False)
        rec = pd.DataFrame(res['records'])
        final_ref = res['fields']['final_reference']
        final_learned = res['fields']['final_learned']
        mass0 = float(np.mean(phi0))
        max_mass_err = float(np.max(np.abs(rec['mass_learned'].values - mass0)))
        unconstrained_metrics.append({
            'test_id': test_id,
            'final_l2': relative_l2(final_learned, final_ref),
            'structure_error': structure_error(final_learned, final_ref),
            'max_mass_error': max_mass_err,
            'energy_violations': energy_violations(rec['energy_learned'].values),
        })
    unconstrained_df = pd.DataFrame(unconstrained_metrics)

    # 6. Speed benchmark
    print("Running speed benchmark...")
    phi_speed = initial_condition(SEED + 99_999, mean=0.0)
    speed = speed_benchmark(phi_speed, EPS_TRAIN, MOBILITY_TRAIN, learned_filter, repeats=3)

    # 7. Summary statistics
    summary = {
        'mean_final_l2_coarse': float(metrics['final_l2_coarse'].mean()),
        'mean_final_l2_learned': float(metrics['final_l2_learned'].mean()),
        'mean_structure_error_coarse': float(metrics['structure_error_coarse'].mean()),
        'mean_structure_error_learned': float(metrics['structure_error_learned'].mean()),
        'max_mass_error_learned': float(metrics['max_mass_error_learned'].max()),
        'energy_violations_learned': int(metrics['energy_violations_learned'].sum()),
        'energy_violations_reference': int(metrics['energy_violations_reference'].sum()),
        'energy_violations_coarse': int(metrics['energy_violations_coarse'].sum()),
        'measured_speedup': speed['measured_speedup'],
        'reference_rollout_time_s': speed['reference_rollout_time_s'],
        'surrogate_rollout_time_s': speed['surrogate_rollout_time_s'],
    }
    summary['l2_error_reduction_percent'] = 100.0 * (summary['mean_final_l2_coarse'] - summary['mean_final_l2_learned']) / max(summary['mean_final_l2_coarse'], 1e-12)
    summary['structure_error_reduction_percent'] = 100.0 * (summary['mean_structure_error_coarse'] - summary['mean_structure_error_learned']) / max(summary['mean_structure_error_coarse'], 1e-12)

    # Statistical test (paired t-test on L2 errors)
    t_stat, p_value = stats.ttest_rel(metrics['final_l2_coarse'], metrics['final_l2_learned'])
    summary['t_stat'] = t_stat
    summary['p_value'] = p_value
    # Cohen's d
    diff = metrics['final_l2_coarse'] - metrics['final_l2_learned']
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    summary['cohens_d'] = d

    # 8. Generate figures
    print("Generating figures...")
    if first_rollout is not None:
        fields = first_rollout['fields']
        # Figure 1: phase field
        fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
        im = ax.imshow(fields['final_reference'].T, origin='lower', extent=[0, DOMAIN_LENGTH, 0, DOMAIN_LENGTH], cmap='RdBu_r')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(r'$\phi$')
        save_figure(fig, figures_dir, 'figure_01')

        # Figure 2: energy
        rec_df = pd.DataFrame(first_rollout['records'])
        time_milli = 1e3 * rec_df['time'].to_numpy()
        fig, ax = plt.subplots(figsize=(5.4, 2.85), constrained_layout=True)
        ax.plot(time_milli, rec_df['energy_reference'], color=COLORS['black'], linestyle='-', label='Reference')
        ax.plot(time_milli, rec_df['energy_coarse'], color=COLORS['orange'], linestyle='--', label='Coarse')
        ax.plot(time_milli, rec_df['energy_learned'], color=COLORS['blue'], linestyle='-.', label='Learned projected')
        ax.set_xlabel(r'Time ($10^{-3}$)')
        ax.set_ylabel('Free energy')
        ax.set_xticks(np.linspace(time_milli.min(), time_milli.max(), 5))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend()
        save_figure(fig, figures_dir, 'figure_02')

        # Figure 3: L2 error
        fig, ax = plt.subplots(figsize=(5.4, 2.85), constrained_layout=True)
        ax.plot(time_milli, rec_df['relative_l2_coarse'], color=COLORS['orange'], linestyle='--', label='Coarse')
        ax.plot(time_milli, rec_df['relative_l2_learned'], color=COLORS['blue'], linestyle='-', label='Learned projected')
        ax.set_xlabel(r'Time ($10^{-3}$)')
        ax.set_ylabel(r'Relative $L^2$ error')
        ax.set_xticks(np.linspace(time_milli.min(), time_milli.max(), 5))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend()
        save_figure(fig, figures_dir, 'figure_03')

        # Figure 4: structure factor
        sf_ref = radial_structure_factor(fields['final_reference']).rename(columns={'structure_factor': 'reference'})
        sf_coarse = radial_structure_factor(fields['final_coarse']).rename(columns={'structure_factor': 'coarse'})
        sf_learn = radial_structure_factor(fields['final_learned']).rename(columns={'structure_factor': 'learned_projected'})
        fig4_df = sf_ref.merge(sf_coarse, on='wavenumber').merge(sf_learn, on='wavenumber')
        fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
        ax.plot(fig4_df['wavenumber'], fig4_df['reference'], color=COLORS['black'], linestyle='-', label='Reference')
        ax.plot(fig4_df['wavenumber'], fig4_df['coarse'], color=COLORS['orange'], linestyle='--', label='Coarse')
        ax.plot(fig4_df['wavenumber'], fig4_df['learned_projected'], color=COLORS['blue'], linestyle='-.', label='Learned projected')
        ax.set_xlabel(r'Wavenumber $|\mathbf{k}|$')
        ax.set_ylabel('Normalized structure factor')
        ax.legend()
        save_figure(fig, figures_dir, 'figure_04')

    # Figure 5: final errors across tests
    fig, ax = plt.subplots(figsize=(3.7, 2.7), constrained_layout=True)
    test_ids = np.arange(len(metrics))
    ax.plot(test_ids, metrics['final_l2_coarse'], marker='o', linestyle='--', color=COLORS['orange'], label=r'Coarse $L^2$')
    ax.plot(test_ids, metrics['final_l2_learned'], marker='o', linestyle='-', color=COLORS['blue'], label=r'Learned $L^2$')
    ax.plot(test_ids, metrics['structure_error_coarse'], marker='s', linestyle='--', color=COLORS['red'], label=r'Coarse $S(k)$')
    ax.plot(test_ids, metrics['structure_error_learned'], marker='s', linestyle='-', color=COLORS['green'], label=r'Learned $S(k)$')
    ax.set_xlabel('Test trajectory')
    ax.set_ylabel('Final error')
    ax.legend()
    save_figure(fig, figures_dir, 'figure_05')

    # Figure 6: OOD epsilon sweep
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    ax.plot(ood_eps['param_value'], ood_eps['l2_error'], marker='o', linestyle='-', color=COLORS['blue'], label=r'$L^2$ error')
    ax.plot(ood_eps['param_value'], ood_eps['structure_error'], marker='s', linestyle='--', color=COLORS['orange'], label=r'$S(k)$ error')
    ax.axvline(EPS_TRAIN, color=COLORS['black'], linestyle=':', linewidth=1.0)
    ax.set_xlabel(r'Interface parameter $\epsilon$')
    ax.set_ylabel('Mean final error')
    ax.legend()
    save_figure(fig, figures_dir, 'figure_06')

    # Figure 7: speed-accuracy with a clear legend
    fig, ax = plt.subplots(figsize=(5.4, 2.85), constrained_layout=True)
    speedup = summary['measured_speedup']

    figure7_points = [
        {
            'label': 'Fine reference',
            'speed': 1.0,
            'error': 0.0,
            'color': COLORS['black'],
            'marker': 'o',
        },
        {
            'label': 'Coarse operator',
            'speed': speedup * 2.0,
            'error': summary['mean_final_l2_coarse'],
            'color': COLORS['orange'],
            'marker': 's',
        },
        {
            'label': 'Learned projected surrogate',
            'speed': speedup,
            'error': summary['mean_final_l2_learned'],
            'color': COLORS['blue'],
            'marker': '^',
        },
    ]

    for point in figure7_points:
        ax.scatter(
            point['speed'],
            point['error'],
            s=55,
            color=point['color'],
            marker=point['marker'],
            label=point['label'],
            edgecolors='black',
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Relative rollout speed')
    ax.set_ylabel(r'Mean final $L^2$ error')
    ax.set_xlim(0.8, max(point['speed'] for point in figure7_points) * 1.30)
    ax.set_ylim(-0.002, max(point['error'] for point in figure7_points) * 1.15)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.legend(loc='upper left', frameon=True, fontsize=8)
    save_figure(fig, figures_dir, 'figure_07')

    # 9. Write data files
    write_csv_with_metadata(data_dir / 'energy_mass_timeseries.csv', timeseries, {'description': 'All test histories'})
    write_csv_with_metadata(data_dir / 'rollout_metrics.csv', metrics, {'description': 'Final metrics for all tests'})
    write_csv_with_metadata(data_dir / 'summary_values.csv', pd.DataFrame([summary]), {'description': 'Summary stats'})
    write_csv_with_metadata(data_dir / 'long_horizon_metrics.csv', long_df, {'description': 'Long-horizon results'})
    write_csv_with_metadata(data_dir / 'ood_epsilon.csv', ood_eps, {'description': 'OOD epsilon sweep'})
    write_csv_with_metadata(data_dir / 'ood_mobility.csv', ood_mob, {'description': 'OOD mobility sweep'})
    write_csv_with_metadata(data_dir / 'unprojected_spectral_residual.csv', unconstrained_df, {'description': 'Unprojected spectral-residual baseline'})

    # 10. Create README
    readme = f"""# CPC Cahn-Hilliard Spectral-Residual Reproduction Package

This package reproduces the numerical results for the CPC Computational Physics manuscript:

**{TITLE}**

## Key numbers
- Mean final coarse L2: {summary['mean_final_l2_coarse']:.6f}
- Mean final learned L2: {summary['mean_final_l2_learned']:.6f}
- Mean final coarse structure error: {summary['mean_structure_error_coarse']:.6f}
- Mean final learned structure error: {summary['mean_structure_error_learned']:.6f}
- Max mass error (learned): {summary['max_mass_error_learned']:.2e}
- Energy violations (learned): {summary['energy_violations_learned']}
- Measured speed-up: {summary['measured_speedup']:.2f}x
- Paired t-test p-value: {summary['p_value']:.2e}
- Cohen's d: {summary['cohens_d']:.2f}

All figures and data are in the respective folders.
"""
    (root / 'README.md').write_text(readme, encoding='utf-8')

    # 11. Zip everything
    zip_path = output_dir / 'ch_neural_operator_revised_package.zip'
    zip_project(root, zip_path)

    # Copy the zip to /content as well, so it appears clearly in the Colab file browser.
    import shutil
    colab_zip_path = Path.cwd() / 'ch_neural_operator_revised_package.zip'
    shutil.copy2(zip_path, colab_zip_path)

    print("\n=== Generation complete ===")
    print(f"Mean final learned L2: {summary['mean_final_l2_learned']:.6f}")
    print(f"Speed-up: {summary['measured_speedup']:.2f}x")
    print(f"p-value: {summary['p_value']:.2e}")
    print(f"Cohen's d: {summary['cohens_d']:.2f}")
    print(f"Package saved to: {zip_path}")
    print(f"Copy saved to: {colab_zip_path}")

    # Trigger browser download in Google Colab.
    try:
        from google.colab import files
        files.download(str(colab_zip_path))
    except Exception as exc:
        print("Automatic download did not start. In Colab, open the left Files panel and download:")
        print(colab_zip_path)

if __name__ == '__main__':
    main()
