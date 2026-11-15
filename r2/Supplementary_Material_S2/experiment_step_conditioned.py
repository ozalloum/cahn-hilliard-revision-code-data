"""Step-conditioned Fourier correction for Cahn-Hilliard simulation.

Compare frozen-field, step-tapered and increment-conditioned corrections
with uncorrected coarse propagators and an independent ETDRK4 reference.

Run from any directory, for example:
    python experiment_step_conditioned.py --train 60 --snapshots 8 --test 10

The training and test initial-condition streams are disjoint. All time-step
variants use one frozen coefficient set fitted at DT0=1e-4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "repro_code" / "numerics" / "validation"))
from kernel import DT, Kernel, legacy_ic, rel, run_etd  # noqa: E402

DT0 = DT
T_FINAL = 0.0035


def historical_mask(n: int) -> np.ndarray:
    """Match the author's original floating-point radius-28 cutoff."""
    wave = np.fft.fftfreq(n, d=2 * np.pi / n) * 2 * np.pi
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    return np.sqrt(kx * kx + ky * ky) <= 28.0


def hermitianize(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    neg = (-np.arange(n)) % n
    return 0.5 * (a + np.conj(a[neg[:, None], neg[None, :]]))


def fit(k: Kernel, trajectories: int, snapshots: int, ridge: float):
    """Fit legacy field and new increment regressions to identical pairs."""
    n = k.n
    numerator_field = np.zeros((n, n), dtype=complex)
    denominator_field = np.zeros((n, n), dtype=float)
    numerator_incr = numerator_field.copy()
    denominator_incr = denominator_field.copy()
    for i in range(trajectories):
        u = legacy_ic(i, train=True)
        for _ in range(snapshots):
            y = k.semi(u, DT0, 100)
            c = k.semi(u, DT0, 6)
            target = np.fft.fft2(y - c)
            field = np.fft.fft2(c)
            incr = np.fft.fft2(c - u)
            numerator_field += target * np.conj(field)
            denominator_field += np.abs(field) ** 2
            numerator_incr += target * np.conj(incr)
            denominator_incr += np.abs(incr) ** 2
            u = y
    r = numerator_field / (denominator_field + ridge)
    b = numerator_incr / (denominator_incr + ridge)
    # Reproduce the historical floating-point wavenumber convention exactly:
    # four axial modes at nominal radius 28 are excluded by roundoff.
    mask = historical_mask(n)
    for w in (r, b):
        w[~mask] = 0
        w[0, 0] = 0
        w[:] = hermitianize(w)
    return r, b


def one_step(k: Kernel, u: np.ndarray, dt: float, name: str,
             r: np.ndarray, b: np.ndarray):
    if name == "C6":
        return k.semi(u, dt, 6), 0
    if name == "C12":
        return k.semi(u, dt, 12), 0
    c = k.semi(u, dt, 6)
    if name == "legacy_fixed":
        feature = np.fft.fft2(c)
        gain = r
    elif name == "field_dt2":
        feature = np.fft.fft2(c)
        gain = r * (dt / DT0) ** 2
    elif name == "increment_dt":
        feature = np.fft.fft2(c - u)
        gain = b * (dt / DT0)
    else:
        raise ValueError(name)
    proposed = c + np.fft.ifft2(gain * feature).real
    accepted, log = k.safeguard(u, proposed)
    return accepted, int(log["code"] > 0)


def evaluate(k: Kernel, ntest: int, dts: list[float],
             r: np.ndarray, b: np.ndarray, out: Path):
    methods = ("C6", "C12", "legacy_fixed", "field_dt2", "increment_dt")
    results = []
    for i in range(ntest):
        u0 = legacy_ic(i, train=False)
        reference = run_etd(k, u0, [T_FINAL], h=DT0 / 8)[T_FINAL]
        if i < 2:
            reference_fine = run_etd(k, u0, [T_FINAL], h=DT0 / 16)[T_FINAL]
            results.append(dict(test_id=i, dt=0, method="reference_refinement",
                                error=rel(reference, reference_fine), backtracks=0,
                                runtime_seconds=0))
        for dt in dts:
            steps_float = T_FINAL / dt
            steps = int(round(steps_float))
            if abs(steps * dt - T_FINAL) > 1e-12:
                raise ValueError("T_FINAL/dt must be an integer")
            for name in methods:
                t0 = time.perf_counter()
                u = u0.copy()
                backtracks = 0
                for _ in range(steps):
                    u, used = one_step(k, u, dt, name, r, b)
                    backtracks += used
                results.append(dict(test_id=i, dt=dt, method=name,
                                    error=rel(u, reference), backtracks=backtracks,
                                    runtime_seconds=time.perf_counter() - t0,
                                    final_mass_error=abs(float(u.mean()-u0.mean())),
                                    final_energy=k.energy(u)))
        print(json.dumps({"completed_test": i + 1, "total": ntest}), flush=True)
    (out / "per_case.json").write_text(json.dumps(results, indent=2))
    summary = []
    for dt in dts:
        for name in methods:
            rows = [x for x in results if x["dt"] == dt and x["method"] == name]
            errors = np.array([x["error"] for x in rows])
            times = np.array([x["runtime_seconds"] for x in rows])
            summary.append(dict(dt=dt, method=name, n=len(rows),
                                error_mean=float(errors.mean()),
                                error_sd=float(errors.std(ddof=1)) if len(rows)>1 else None,
                                mean_seconds=float(times.mean()),
                                backtracking_total=sum(x["backtracks"] for x in rows)))
    refines = [x["error"] for x in results if x["method"] == "reference_refinement"]
    (out / "summary.json").write_text(json.dumps(dict(summary=summary,
            max_reference_refinement=max(refines) if refines else None), indent=2))
    print(json.dumps(dict(summary=summary,
        max_reference_refinement=max(refines) if refines else None), indent=2))


def verify_increment_properties(k: Kernel, b: np.ndarray, out: Path):
    """Check the proved local scaling without the acceptance safeguard."""
    u = legacy_ic(0, train=False)
    hs = [DT0 / (2 ** j) for j in range(7)]
    norms = []
    mass_drift = []
    for h in hs:
        c = k.semi(u, h, 6)
        corr = np.fft.ifft2((h / DT0) * b *
                            np.fft.fft2(c - u)).real
        norms.append(float(np.sqrt(np.mean(corr * corr))))
        mass_drift.append(abs(float(corr.mean())))
    orders = [float(np.log2(a / z)) for a, z
              in zip(norms[:-1], norms[1:])]
    fixed_point_changes = {}
    for constant in (0.0, 1.0):
        v = np.full_like(u, constant)
        c = k.semi(v, DT0, 6)
        w = c + np.fft.ifft2(b * np.fft.fft2(c - v)).real
        fixed_point_changes[str(constant)] = float(np.max(np.abs(w - v)))
    result = dict(hs=hs, correction_norms=norms,
                  apparent_orders=orders, max_correction_mass_drift=max(mass_drift),
                  fixed_point_changes=fixed_point_changes)
    (out / "structural_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(structural_checks=result)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--dts", nargs="+", type=float,
                        default=[DT0, DT0 / 2, DT0 / 4, DT0 / 8])
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--reuse-filters", type=Path,
                        help="Use saved R and B from an earlier full fit")
    parser.add_argument("--output", type=Path, default=ROOT / "step_conditioned_results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    k = Kernel()
    if args.reuse_filters:
        loaded = np.load(args.reuse_filters)
        r, b = loaded["R"].copy(), loaded["B"].copy()
        mask = historical_mask(k.n)
        for coef in (r, b):
            coef[~mask] = 0
            coef[0, 0] = 0
        train_seconds = None
    else:
        t0 = time.perf_counter()
        r, b = fit(k, args.train, args.snapshots, args.ridge)
        train_seconds = time.perf_counter() - t0
    np.savez_compressed(args.output / "exploratory_filters.npz", R=r, B=b,
                        train_trajectories=args.train, snapshots=args.snapshots,
                        dt0=DT0, ridge=args.ridge)
    meta = dict(train_trajectories=args.train, snapshots=args.snapshots,
                dt_training=DT0, training_seconds=train_seconds,
                numpy_version=np.__version__,
                peak_R=float(np.abs(r).max()), peak_B=float(np.abs(b).max()),
                method_status="Step-conditioned Fourier correction")
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    verify_increment_properties(k, b, args.output)
    evaluate(k, args.test, args.dts, r, b, args.output)


if __name__ == "__main__":
    main()
