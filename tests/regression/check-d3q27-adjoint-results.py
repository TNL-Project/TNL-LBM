"""Check D3Q27 adjoint LBM workflow results.

The adjoint workflow has two stages:
  1. sim_pseudomeasure — generates synthetic "measured" data
  2. sim_adjoint — runs primary + adjoint, iterates to minimize loss

This script only reads existing outputs — it does not launch any
simulation. It checks:

  Pseudomeasure:
    - 3D BP5 fields are finite
    - Mass conservation (steady-state channel flow)
    - Wall / inflow boundary cells have expected velocity
    - Measured macro BP5 file exists and is readable

  Adjoint:
    - Loss function file exists, values are finite and positive
    - Loss is monotonically non-increasing across epochs
    - Velocity profile files (X, Y, Z) are finite
    - Primary and adjoint 3D BP5 fields are finite (if present)
"""

import argparse
import glob
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

import adios2
import numpy as np

# D3Q27 GEO enum (subset relevant to adjoint sims)
GEO_FLUID = 0
GEO_WALL = 1
GEO_INFLOW_BB_LEFT = 4


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"  [{status}] {self.name}"
        if self.detail:
            msg += f": {self.detail}"
        return msg


@dataclass
class SimReport:
    sim_name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        lines = [f"{self.sim_name} [{status}]"]
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)


def read_last_step(
    bp_path: pathlib.Path, var_names: list[str]
) -> dict[str, np.ndarray]:
    if not bp_path.exists():
        raise FileNotFoundError(f"Output file does not exist: {bp_path}")

    data: dict[str, np.ndarray] = {}
    with adios2.FileReader(str(bp_path)) as reader:
        num_steps = reader.num_steps()
        if num_steps == 0:
            raise RuntimeError(f"No steps written to {bp_path}")

        available = reader.available_variables()
        missing = [v for v in var_names if v not in available]
        if missing:
            raise RuntimeError(f"Missing variables in {bp_path}: {missing}")

        for name in var_names:
            var = reader.inquire_variable(name)
            arr = reader.read(var, step_selection=[num_steps - 1, 1])
            data[name] = np.asarray(arr)

    return data


def check_finite(data: dict[str, np.ndarray]) -> CheckResult:
    for name, arr in data.items():
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            return CheckResult(
                "finiteness", False, f"{name} has {n_bad} non-finite values"
            )
    return CheckResult("finiteness", True, f"all {len(data)} fields finite")


def check_pseudomeasure(base: pathlib.Path) -> SimReport:
    report = SimReport("pseudomeasure")

    # 3D BP5 output
    bp3d = base / "results_sim_pseudomeasure_res01_np001" / "output_3D.bp"
    try:
        data = read_last_step(
            bp3d,
            [
                "lbm_density",
                "lbm_velocity_x",
                "lbm_velocity_y",
                "lbm_velocity_z",
                "wall",
            ],
        )
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read_3d", False, str(exc)))
        return report

    rho = data["lbm_density"]
    vx = data["lbm_velocity_x"]
    vy = data["lbm_velocity_y"]
    vz = data["lbm_velocity_z"]
    wall = data["wall"]

    report.results.append(check_finite(data))

    # Mass conservation (steady-state, should be very close to 1)
    mean_rho = float(np.mean(rho))
    dev = abs(mean_rho - 1.0)
    report.results.append(
        CheckResult("mass", dev < 1e-3, f"mean(rho)={mean_rho:.8e}, |mean-1|={dev:.2e}")
    )

    # Wall no-slip
    solid_mask = wall == GEO_WALL
    if solid_mask.any():
        max_v = max(
            float(np.max(np.abs(vx[solid_mask]))),
            float(np.max(np.abs(vy[solid_mask]))),
            float(np.max(np.abs(vz[solid_mask]))),
        )
        report.results.append(
            CheckResult(
                "wall_no_slip", max_v < 1e-10, f"max|v| in wall cells={max_v:.2e}"
            )
        )

    # Inflow present (GEO_INFLOW_BB_LEFT = 4)
    inflow_mask = wall == GEO_INFLOW_BB_LEFT
    report.results.append(
        CheckResult(
            "inflow_present",
            inflow_mask.any(),
            f"{int(inflow_mask.sum())} inflow cells",
        )
    )

    # Measured macro file exists
    measured_bp = (
        base / "results_adjoint" / "adjoint_data_res01" / "macro_measured.bp.bp"
    )
    report.results.append(
        CheckResult("measured_file", measured_bp.exists(), str(measured_bp.name))
    )

    return report


def check_adjoint(base: pathlib.Path) -> SimReport:
    report = SimReport("adjoint")

    adjoint_dir = base / "results_adjoint" / "adjoint_data_res01"

    # Loss function file
    loss_file = adjoint_dir / "lossFunction.txt"
    if not loss_file.exists():
        report.results.append(CheckResult("loss_file", False, f"{loss_file} not found"))
        return report

    try:
        loss_values = [
            float(line.strip()) for line in loss_file.read_text().strip().splitlines()
        ]
    except Exception as exc:
        report.results.append(CheckResult("loss_file", False, str(exc)))
        return report

    report.results.append(
        CheckResult(
            "loss_file",
            len(loss_values) >= 1,
            f"{len(loss_values)} epoch(s) recorded",
        )
    )

    # All loss values finite and positive
    all_finite = all(np.isfinite(v) for v in loss_values)
    all_positive = all(v > 0 for v in loss_values)
    report.results.append(
        CheckResult(
            "loss_finite_positive",
            all_finite and all_positive,
            f"values={['%.6e' % v for v in loss_values]}",
        )
    )

    # Loss monotonically non-increasing (skipped if only 1 epoch)
    if len(loss_values) >= 2:
        monotonic = all(
            loss_values[i] <= loss_values[i - 1] for i in range(1, len(loss_values))
        )
        report.results.append(
            CheckResult("loss_monotonic", monotonic, f"monotonic={monotonic}")
        )

    # Velocity profiles (X, Y, Z)
    for axis in ["X", "Y", "Z"]:
        profile_file = adjoint_dir / f"velocityProfile{axis}.txt"
        if not profile_file.exists():
            report.results.append(
                CheckResult(
                    f"velocity_profile_{axis}", False, f"{profile_file.name} not found"
                )
            )
            continue
        try:
            profile = np.loadtxt(profile_file)
            n_nan = int(np.sum(~np.isfinite(profile)))
            report.results.append(
                CheckResult(
                    f"velocity_profile_{axis}",
                    n_nan == 0,
                    f"shape={profile.shape}, NaN count={n_nan}, "
                    f"range=[{np.nanmin(profile):.4e}, {np.nanmax(profile):.4e}]",
                )
            )
        except Exception as exc:
            report.results.append(
                CheckResult(f"velocity_profile_{axis}", False, str(exc))
            )

    # Primary simulation 3D output (if present)
    primary_dirs = sorted(glob.glob(str(base / "results_sim_primary_*")))
    if primary_dirs:
        bp3d = pathlib.Path(primary_dirs[0]) / "output_3D.bp"
        if bp3d.exists():
            try:
                data = read_last_step(
                    bp3d,
                    [
                        "lbm_density",
                        "lbm_velocity_x",
                        "lbm_velocity_y",
                        "lbm_velocity_z",
                    ],
                )
                report.results.append(check_finite(data))

                rho = data["lbm_density"]
                mean_rho = float(np.mean(rho))
                dev = abs(mean_rho - 1.0)
                report.results.append(
                    CheckResult(
                        "primary_mass",
                        dev < 5e-3,
                        f"mean(rho)={mean_rho:.8e}, |mean-1|={dev:.2e}",
                    )
                )
            except (FileNotFoundError, RuntimeError) as exc:
                report.results.append(CheckResult("primary_3d", False, str(exc)))

    # Adjoint simulation 3D output (if present)
    adjoint_dirs = sorted(glob.glob(str(base / "results_sim_adjoint_*")))
    if adjoint_dirs:
        bp3d = pathlib.Path(adjoint_dirs[0]) / "output_3D.bp"
        if bp3d.exists():
            try:
                data = read_last_step(
                    bp3d,
                    [
                        "lbm_density",
                        "lbm_velocity_x",
                        "lbm_velocity_y",
                        "lbm_velocity_z",
                    ],
                )
                report.results.append(check_finite(data))
            except (FileNotFoundError, RuntimeError) as exc:
                report.results.append(CheckResult("adjoint_3d", False, str(exc)))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check D3Q27 adjoint LBM workflow results from existing outputs."
    )
    parser.add_argument("--base", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()

    reports = [check_pseudomeasure(args.base), check_adjoint(args.base)]

    for report in reports:
        print(report.summary())
        print()

    total = len(reports)
    passed = sum(1 for r in reports if r.all_passed)
    print(f"{passed}/{total} stages passed all checks.")

    if passed < total:
        print("\nFailures:")
        for report in reports:
            if not report.all_passed:
                for r in report.results:
                    if not r.passed:
                        print(f"  {report.sim_name}: {r.name}: {r.detail}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
